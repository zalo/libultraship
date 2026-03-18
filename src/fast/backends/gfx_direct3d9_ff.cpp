// gfx_direct3d9_ff.cpp — D3D9 fixed-function rendering backend for RTX Remix integration.
//
// This backend intentionally uses the D3D9 fixed-function pipeline (not programmable shaders) so that
// RTX Remix (dxvk-remix) can intercept the draw calls and reconstruct the 3D scene for path tracing.
//
// Integration notes:
//   • Place d3d9.dll (the RTX Remix interposer) next to the executable.
//   • Place the .trex/ runtime directory next to the executable.
//   • Select backend id 3 (FAST3D_DXGI_DX9) in the config file:  Window.Backend.Id = 3
//
// Vertex format emitted per triangle (12 floats per vertex, 36 floats per triangle):
//   [wx wy wz]  — world-space position (after modelview, before projection)
//   [nx ny nz]  — object-space surface normal (normalised to [-1,1])
//   [r  g  b a] — diffuse RGBA (0-1 floats; packed to D3DCOLOR in DrawTriangles)
//   [u  v]      — texture UV (normalised 0-1 for tile 0)
//
// Scene lights are forwarded to D3D9 via SetLight() every time the N64 lights change.

#ifdef ENABLE_DX9

#pragma comment(lib, "d3d9.lib")

#include <cmath>
#include <cstring>
#include <algorithm>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>

#include "fast/backends/gfx_direct3d9_ff.h"
#include "fast/backends/gfx_dxgi.h"
#include "fast/interpreter.h"
#include "ship/Context.h"
#include "ship/window/gui/Gui.h"
#include "spdlog/spdlog.h"
#include <imgui_impl_dx9.h>
#include <imgui_impl_win32.h>

// N64 GBI constants needed for light type identification
#ifndef G_LIGHTING_POSITIONAL
#define G_LIGHTING_POSITIONAL 0x00400000
#endif

namespace Fast {

// ---------------------------------------------------------------------------
// Vertex layout (matches what the interpreter emits when OwnsLighting()==true)
// ---------------------------------------------------------------------------
#define D3D9FF_VBO_FLOATS_PER_VERTEX 12 // wx wy wz nx ny nz r g b a u v

struct VertexD3D9FF {
    float x, y, z;     // World-space position
    float nx, ny, nz;  // Surface normal
    DWORD diffuse;      // Packed D3DCOLOR (ARGB)
    float u, v;         // Texture UV
};

#define D3D9FF_FVF (D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1)
static_assert(sizeof(VertexD3D9FF) == 36, "VertexD3D9FF size mismatch");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline D3DCOLOR FloatRGBAtoD3DCOLOR(float r, float g, float b, float a) {
    auto clamp01 = [](float v) -> BYTE { return (BYTE)(std::max(0.0f, std::min(1.0f, v)) * 255.0f + 0.5f); };
    return D3DCOLOR_ARGB(clamp01(a), clamp01(r), clamp01(g), clamp01(b));
}

static inline void SetSamplerAddressMode(IDirect3DDevice9* dev, DWORD slot, DWORD uOrV, uint32_t cm) {
    DWORD mode;
    if (cm & 2 /*G_TX_MIRROR*/) mode = D3DTADDRESS_MIRROR;
    else if (cm & 1 /*G_TX_CLAMP*/) mode = D3DTADDRESS_CLAMP;
    else mode = D3DTADDRESS_WRAP;
    dev->SetSamplerState(slot, uOrV, mode);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
GfxRenderingAPIDX9FF::GfxRenderingAPIDX9FF(GfxWindowBackendDXGI* windowBackend)
    : mWindowBackend(windowBackend) {}

GfxRenderingAPIDX9FF::~GfxRenderingAPIDX9FF() {
    if (mVertexBuffer) { mVertexBuffer->Release(); mVertexBuffer = nullptr; }
    for (auto& tex : mTextures) {
        if (tex.texture) { tex.texture->Release(); tex.texture = nullptr; }
    }
    for (auto& fb : mFrameBuffers) {
        if (fb.colorSurface) { fb.colorSurface->Release(); fb.colorSurface = nullptr; }
        if (fb.depthSurface) { fb.depthSurface->Release(); fb.depthSurface = nullptr; }
    }
    if (mDevice)    { mDevice->Release();    mDevice    = nullptr; }
    if (mD3D)       { mD3D->Release();       mD3D       = nullptr; }
    if (mD3D9Module) { FreeLibrary(mD3D9Module); mD3D9Module = nullptr; }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::Init() {
    SPDLOG_INFO("[DX9FF] Initialising D3D9 fixed-function backend (RTX Remix target)");

    // Load d3d9.dll — RTX Remix replaces this DLL in the game directory, so the
    // interposer is transparently loaded here instead of the system DLL.
    mD3D9Module = LoadLibraryW(L"d3d9.dll");
    if (!mD3D9Module) {
        MessageBoxA(nullptr, "d3d9.dll not found.", "DX9FF Error", MB_OK | MB_ICONERROR);
        return;
    }

    auto Direct3DCreate9Fn = reinterpret_cast<IDirect3D9* (WINAPI*)(UINT)>(
        GetProcAddress(mD3D9Module, "Direct3DCreate9"));
    if (!Direct3DCreate9Fn) {
        MessageBoxA(nullptr, "Direct3DCreate9 not found in d3d9.dll.", "DX9FF Error", MB_OK | MB_ICONERROR);
        return;
    }

    mD3D = Direct3DCreate9Fn(D3D_SDK_VERSION);
    if (!mD3D) {
        MessageBoxA(nullptr, "Direct3DCreate9 failed.", "DX9FF Error", MB_OK | MB_ICONERROR);
        return;
    }

    HWND hwnd = mWindowBackend->GetWindowHandle();

    RECT rc;
    GetClientRect(hwnd, &rc);
    uint32_t w = rc.right  - rc.left;
    uint32_t h = rc.bottom - rc.top;
    mRenderTargetHeight = (int32_t)h;

    D3DPRESENT_PARAMETERS pp = {};
    pp.Windowed             = TRUE;
    pp.SwapEffect           = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferFormat     = D3DFMT_X8R8G8B8;
    pp.BackBufferWidth      = w;
    pp.BackBufferHeight     = h;
    pp.EnableAutoDepthStencil = TRUE;
    pp.AutoDepthStencilFormat = D3DFMT_D24S8;
    pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    pp.hDeviceWindow        = hwnd;

    HRESULT hr = mD3D->CreateDevice(
        D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hwnd,
        D3DCREATE_HARDWARE_VERTEXPROCESSING,
        &pp, &mDevice);
    if (FAILED(hr)) {
        char msg[256];
        sprintf_s(msg, "CreateDevice failed. HRESULT: 0x%08X", (unsigned)hr);
        MessageBoxA(hwnd, msg, "DX9FF Error", MB_OK | MB_ICONERROR);
        return;
    }

    ResetDeviceState();

    // Create dynamic vertex buffer sized to hold one full MAX_TRI_BUFFER batch.
    // MAX_TRI_BUFFER == 256 (interpreter.h). 256 tris * 3 verts * 36 bytes.
    UINT vbSize = 256 * 3 * sizeof(VertexD3D9FF);
    hr = mDevice->CreateVertexBuffer(vbSize, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                                     D3D9FF_FVF, D3DPOOL_DEFAULT, &mVertexBuffer, nullptr);
    if (FAILED(hr)) {
        SPDLOG_ERROR("[DX9FF] CreateVertexBuffer failed: 0x{:08X}", (unsigned)hr);
    }

    // Create the default framebuffer (represents the swap-chain back-buffer).
    CreateFramebuffer();

    SPDLOG_INFO("[DX9FF] D3D9 device created ({0}x{1})", w, h);

    // Notify the GUI system so it can initialise ImGui with our D3D9 device.
    Ship::GuiWindowInitData windowImpl;
    windowImpl.Dx11 = { hwnd, nullptr, mDevice };
    Ship::Context::GetInstance()->GetWindow()->GetGui()->Init(windowImpl);
}

// ---------------------------------------------------------------------------
// ResetDeviceState — one-time render states that match N64 rendering behaviour
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::ResetDeviceState() {
    if (!mDevice) return;

    // Enable T&L lighting so SetLight() calls are honoured by RTX Remix.
    mDevice->SetRenderState(D3DRS_LIGHTING,              TRUE);
    mDevice->SetRenderState(D3DRS_NORMALIZENORMALS,      TRUE);
    mDevice->SetRenderState(D3DRS_SPECULARENABLE,        FALSE);
    mDevice->SetRenderState(D3DRS_AMBIENT,               D3DCOLOR_ARGB(255, 0, 0, 0));

    // Vertex colour source — diffuse from vertex, ambient from material.
    mDevice->SetRenderState(D3DRS_COLORVERTEX,           TRUE);
    mDevice->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_COLOR1);
    mDevice->SetRenderState(D3DRS_AMBIENTMATERIALSOURCE, D3DMCS_MATERIAL);

    // Alpha blending — set up src/dst once; enable/disable per draw.
    mDevice->SetRenderState(D3DRS_SRCBLEND,              D3DBLEND_SRCALPHA);
    mDevice->SetRenderState(D3DRS_DESTBLEND,             D3DBLEND_INVSRCALPHA);
    mDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,      FALSE);

    // Alpha testing — disabled by default.
    mDevice->SetRenderState(D3DRS_ALPHATESTENABLE,       FALSE);
    mDevice->SetRenderState(D3DRS_ALPHAFUNC,             D3DCMP_GREATEREQUAL);
    mDevice->SetRenderState(D3DRS_ALPHAREF,              0x80);

    // Depth
    mDevice->SetRenderState(D3DRS_ZENABLE,               D3DZB_TRUE);
    mDevice->SetRenderState(D3DRS_ZWRITEENABLE,          TRUE);
    mDevice->SetRenderState(D3DRS_ZFUNC,                 D3DCMP_LESSEQUAL);

    // Culling — N64 geometry is pre-culled by the interpreter; disable D3D9 culling.
    mDevice->SetRenderState(D3DRS_CULLMODE,              D3DCULL_NONE);

    // Texture stage 1 disabled by default.
    mDevice->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
    mDevice->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);

    // Identity transforms — the interpreter provides world-space positions already.
    D3DMATRIX identity = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    mDevice->SetTransform(D3DTS_WORLD, &identity);
    mDevice->SetTransform(D3DTS_VIEW,  &identity);
    mDevice->SetTransform(D3DTS_PROJECTION, &identity);
}

// ---------------------------------------------------------------------------
// Frame control
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::StartFrame() {
    if (mDevice) mDevice->BeginScene();
}

void GfxRenderingAPIDX9FF::EndFrame() {
    if (mDevice) {
        mDevice->EndScene();
        mDevice->Present(nullptr, nullptr, nullptr, nullptr);
    }
}

void GfxRenderingAPIDX9FF::FinishRender() {}

void GfxRenderingAPIDX9FF::OnResize() {
    if (!mDevice || !mWindowBackend) return;

    HWND hwnd = mWindowBackend->GetWindowHandle();
    RECT rc;
    GetClientRect(hwnd, &rc);
    uint32_t w = rc.right  - rc.left;
    uint32_t h = rc.bottom - rc.top;
    mRenderTargetHeight = (int32_t)h;

    // Release resources that are tied to the swap chain size.
    if (mVertexBuffer) { mVertexBuffer->Release(); mVertexBuffer = nullptr; }
    if (!mFrameBuffers.empty()) {
        auto& fb0 = mFrameBuffers[0];
        if (fb0.colorSurface) { fb0.colorSurface->Release(); fb0.colorSurface = nullptr; }
        if (fb0.depthSurface) { fb0.depthSurface->Release(); fb0.depthSurface = nullptr; }
    }

    D3DPRESENT_PARAMETERS pp = {};
    pp.Windowed             = TRUE;
    pp.SwapEffect           = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferFormat     = D3DFMT_X8R8G8B8;
    pp.BackBufferWidth      = w;
    pp.BackBufferHeight     = h;
    pp.EnableAutoDepthStencil = TRUE;
    pp.AutoDepthStencilFormat = D3DFMT_D24S8;
    pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;

    HRESULT hr = mDevice->Reset(&pp);
    if (FAILED(hr)) {
        SPDLOG_ERROR("[DX9FF] Device Reset failed: 0x{:08X}", (unsigned)hr);
        return;
    }

    ResetDeviceState();

    UINT vbSize = 256 * 3 * sizeof(VertexD3D9FF);
    mDevice->CreateVertexBuffer(vbSize, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                                D3D9FF_FVF, D3DPOOL_DEFAULT, &mVertexBuffer, nullptr);
}

// ---------------------------------------------------------------------------
// CommitProjection — keep D3D9's projection matrix in sync with the RSP P_matrix
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::CommitProjection(const float pMatrix[4][4]) {
    if (!mDevice) return;
    // Cache and only call SetTransform when the matrix actually changes.
    if (!mPMatrixDirty && memcmp(mCachedPMatrix, pMatrix, sizeof(mCachedPMatrix)) == 0) return;
    memcpy(mCachedPMatrix, pMatrix, sizeof(mCachedPMatrix));
    mPMatrixDirty = false;
    // The RSP P_matrix transforms camera-space → clip-space.
    // We emit world_pos (= modelview * ob, i.e. camera-space) as vertex positions,
    // so setting D3DTS_PROJECTION to P_matrix gives us correct clip-space output.
    D3DMATRIX d3dProj;
    memcpy(&d3dProj, pMatrix, sizeof(D3DMATRIX));
    mDevice->SetTransform(D3DTS_PROJECTION, &d3dProj);
}

// ---------------------------------------------------------------------------
// CommitLights — forward N64 scene lights to D3D9 so RTX Remix can path-trace them
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::CommitLights(const RSP* rsp) {
    if (!mDevice || !rsp) return;

    int numSceneLights = rsp->current_num_lights - 1; // last entry is ambient

    // Ambient light
    const auto& amb = rsp->current_lights[rsp->current_num_lights - 1].l;
    D3DCOLOR ambColor = D3DCOLOR_COLORVALUE(amb.col[0] / 255.0f,
                                            amb.col[1] / 255.0f,
                                            amb.col[2] / 255.0f, 1.0f);
    mDevice->SetRenderState(D3DRS_AMBIENT, ambColor);

    // Directional / positional scene lights
    for (int i = 0; i < numSceneLights; i++) {
        const F3DLight& fl = rsp->current_lights[i];
        bool isPositional = (fl.p.unk3 != 0);

        D3DLIGHT9 light = {};

        if (isPositional) {
            light.Type      = D3DLIGHT_POINT;
            light.Position  = { (float)fl.p.pos[0], (float)fl.p.pos[1], (float)fl.p.pos[2] };
            light.Range     = 65535.0f;
            // Map N64 attenuation coefficients (same formula as interpreter.cpp lines 1307-1312)
            light.Attenuation0 = 1.0f;
            light.Attenuation1 = fl.p.unk7 * 2.0f / 65535.0f;
            light.Attenuation2 = fl.p.unkE / (8.0f * 65535.0f);
        } else {
            // Directional light. The precomputed coefficients are in camera space;
            // use the raw direction bytes for an approximate world-space direction.
            light.Type      = D3DLIGHT_DIRECTIONAL;
            float dx = fl.l.dir[0] / 127.0f;
            float dy = fl.l.dir[1] / 127.0f;
            float dz = fl.l.dir[2] / 127.0f;
            float len = sqrtf(dx*dx + dy*dy + dz*dz);
            if (len < 0.001f) len = 1.0f;
            light.Direction = { dx/len, dy/len, dz/len };
        }

        light.Diffuse.r = fl.l.col[0] / 255.0f;
        light.Diffuse.g = fl.l.col[1] / 255.0f;
        light.Diffuse.b = fl.l.col[2] / 255.0f;
        light.Diffuse.a = 1.0f;

        mDevice->SetLight(i, &light);
        mDevice->LightEnable(i, TRUE);
    }

    // Disable any lights left over from a previous frame that had more lights.
    for (int i = numSceneLights; i < mPrevLightCount; i++) {
        mDevice->LightEnable(i, FALSE);
    }
    mPrevLightCount = numSceneLights;
}

// ---------------------------------------------------------------------------
// Shader system (maps N64 color combiners to D3D9 texture stage states)
// ---------------------------------------------------------------------------
TSSConfig GfxRenderingAPIDX9FF::BuildTSSFromCC(uint64_t shaderId0, uint32_t shaderId1,
                                               bool& outAlpha, bool outUsedTex[2]) {
    CCFeatures cc = {};
    gfx_cc_get_features(shaderId0, shaderId1, &cc);

    outAlpha       = cc.opt_alpha;
    outUsedTex[0]  = cc.usedTextures[0];
    outUsedTex[1]  = cc.usedTextures[1];

    TSSConfig tss = {};
    tss.useStage1         = false;
    tss.setTextureFactor  = false;

    // Stage 0 — choose the operation based on what the combiner uses.
    if (cc.usedTextures[0] && (cc.do_single[0][0] || cc.do_multiply[0][0] || cc.do_mix[0][0])) {
        if (cc.do_single[0][0]) {
            // Only texture, no shade modulation.
            tss.colorOp0   = D3DTOP_SELECTARG1;
            tss.colorArg1_0 = D3DTA_TEXTURE;
            tss.colorArg2_0 = D3DTA_DIFFUSE;
            tss.alphaOp0   = D3DTOP_SELECTARG1;
            tss.alphaArg1_0 = D3DTA_TEXTURE;
            tss.alphaArg2_0 = D3DTA_DIFFUSE;
        } else {
            // Texture modulated by vertex diffuse (most common: TEXEL0 * SHADE).
            tss.colorOp0    = D3DTOP_MODULATE;
            tss.colorArg1_0 = D3DTA_TEXTURE;
            tss.colorArg2_0 = D3DTA_DIFFUSE;
            tss.alphaOp0    = D3DTOP_MODULATE;
            tss.alphaArg1_0 = D3DTA_TEXTURE;
            tss.alphaArg2_0 = D3DTA_DIFFUSE;
        }
    } else {
        // No texture or untextured — pass through vertex diffuse.
        tss.colorOp0    = D3DTOP_SELECTARG1;
        tss.colorArg1_0 = D3DTA_DIFFUSE;
        tss.colorArg2_0 = D3DTA_DIFFUSE;
        tss.alphaOp0    = D3DTOP_SELECTARG1;
        tss.alphaArg1_0 = D3DTA_DIFFUSE;
        tss.alphaArg2_0 = D3DTA_DIFFUSE;
    }

    // Stage 1 — two-cycle combiner: modulate stage 0 result with texture 1.
    if (cc.opt_2cyc && cc.usedTextures[1]) {
        tss.useStage1    = true;
        tss.colorOp1     = D3DTOP_MODULATE;
        tss.colorArg1_1  = D3DTA_TEXTURE;
        tss.colorArg2_1  = D3DTA_CURRENT;
        tss.alphaOp1     = D3DTOP_MODULATE;
        tss.alphaArg1_1  = D3DTA_TEXTURE;
        tss.alphaArg2_1  = D3DTA_CURRENT;
    }

    return tss;
}

ShaderProgram* GfxRenderingAPIDX9FF::CreateAndLoadNewShader(uint64_t shaderId0, uint32_t shaderId1) {
    auto key = std::make_pair(shaderId0, (uint32_t)shaderId1);
    ShaderProgramD3D9FF& prog = mShaderPool[key];
    prog.shaderId0  = shaderId0;
    prog.shaderId1  = shaderId1;
    bool outUsed[2] = {};
    prog.tss        = BuildTSSFromCC(shaderId0, shaderId1, prog.optAlpha, outUsed);
    prog.usedTextures[0] = outUsed[0];
    prog.usedTextures[1] = outUsed[1];
    prog.numInputs  = 0;
    mCurrentProgram = &prog;
    return &prog;
}

ShaderProgram* GfxRenderingAPIDX9FF::LookupShader(uint64_t shaderId0, uint32_t shaderId1) {
    auto it = mShaderPool.find(std::make_pair(shaderId0, (uint32_t)shaderId1));
    return (it != mShaderPool.end()) ? &it->second : nullptr;
}

void GfxRenderingAPIDX9FF::LoadShader(ShaderProgram* newPrg) {
    mCurrentProgram = static_cast<ShaderProgramD3D9FF*>(newPrg);
}

void GfxRenderingAPIDX9FF::UnloadShader(ShaderProgram*) {}

void GfxRenderingAPIDX9FF::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    auto* p = static_cast<ShaderProgramD3D9FF*>(prg);
    *numInputs       = p ? p->numInputs       : 0;
    usedTextures[0]  = p ? p->usedTextures[0] : false;
    usedTextures[1]  = p ? p->usedTextures[1] : false;
}

// ---------------------------------------------------------------------------
// Texture management
// ---------------------------------------------------------------------------
uint32_t GfxRenderingAPIDX9FF::NewTexture() {
    mTextures.push_back({});
    return (uint32_t)(mTextures.size() - 1);
}

void GfxRenderingAPIDX9FF::SelectTexture(int tile, uint32_t textureId) {
    if (tile < 2) mCurrentTextureIds[tile] = textureId;
}

void GfxRenderingAPIDX9FF::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    if (!mDevice) return;

    uint32_t id = mCurrentTextureIds[0];
    if (id >= mTextures.size()) return;

    TextureD3D9& tex = mTextures[id];
    if (tex.texture) { tex.texture->Release(); tex.texture = nullptr; }

    HRESULT hr = mDevice->CreateTexture(width, height, 1, 0,
                                        D3DFMT_A8R8G8B8, D3DPOOL_MANAGED,
                                        &tex.texture, nullptr);
    if (FAILED(hr)) {
        SPDLOG_WARN("[DX9FF] CreateTexture failed: 0x{:08X}", (unsigned)hr);
        return;
    }

    D3DLOCKED_RECT lr;
    hr = tex.texture->LockRect(0, &lr, nullptr, 0);
    if (FAILED(hr)) return;

    // RGBA→BGRA conversion (D3DFMT_A8R8G8B8 is stored as BGRA in memory).
    auto* dst = static_cast<uint8_t*>(lr.pBits);
    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = dst + y * lr.Pitch;
        const uint8_t* src = rgba32Buf + y * width * 4;
        for (uint32_t x = 0; x < width; x++) {
            row[x*4+0] = src[x*4+2]; // B
            row[x*4+1] = src[x*4+1]; // G
            row[x*4+2] = src[x*4+0]; // R
            row[x*4+3] = src[x*4+3]; // A
        }
    }
    tex.texture->UnlockRect(0);

    tex.width  = width;
    tex.height = height;
}

void GfxRenderingAPIDX9FF::SetSamplerParameters(int sampler, bool linearFilter, uint32_t cms, uint32_t cmt) {
    if (!mDevice) return;
    DWORD slot = (DWORD)sampler;
    D3DTEXTUREFILTERTYPE filter = linearFilter ? D3DTEXF_LINEAR : D3DTEXF_POINT;
    mDevice->SetSamplerState(slot, D3DSAMP_MINFILTER, filter);
    mDevice->SetSamplerState(slot, D3DSAMP_MAGFILTER, filter);
    mDevice->SetSamplerState(slot, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    SetSamplerAddressMode(mDevice, slot, D3DSAMP_ADDRESSU, cms);
    SetSamplerAddressMode(mDevice, slot, D3DSAMP_ADDRESSV, cmt);

    if (sampler < 2) {
        auto& t = mTextures[mCurrentTextureIds[sampler]];
        t.linearFiltering = linearFilter;
        t.cms = (uint8_t)cms;
        t.cmt = (uint8_t)cmt;
    }
}

void GfxRenderingAPIDX9FF::DeleteTexture(uint32_t texId) {
    if (texId >= mTextures.size()) return;
    auto& t = mTextures[texId];
    if (t.texture) { t.texture->Release(); t.texture = nullptr; }
}

void GfxRenderingAPIDX9FF::SelectTextureFb(int fbId) {
    if (fbId >= 0 && fbId < (int)mFrameBuffers.size()) {
        SelectTexture(0, mFrameBuffers[fbId].textureId);
    }
}

// ---------------------------------------------------------------------------
// Render-state helpers
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::SetDepthTestAndMask(bool depthTest, bool zUpd) {
    if (!mDevice) return;
    mDevice->SetRenderState(D3DRS_ZENABLE,      depthTest ? D3DZB_TRUE : D3DZB_FALSE);
    mDevice->SetRenderState(D3DRS_ZWRITEENABLE, zUpd      ? TRUE       : FALSE);
}

void GfxRenderingAPIDX9FF::SetZmodeDecal(bool decal) {
    if (!mDevice) return;
    float bias = decal ? -2.0f : 0.0f;
    DWORD biasInt;
    memcpy(&biasInt, &bias, sizeof(float));
    mDevice->SetRenderState(D3DRS_DEPTHBIAS,         biasInt);
    mDevice->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, biasInt);
}

void GfxRenderingAPIDX9FF::SetViewport(int x, int y, int width, int height) {
    if (!mDevice) return;
    D3DVIEWPORT9 vp;
    vp.X      = (DWORD)x;
    vp.Y      = (DWORD)(mRenderTargetHeight - y - height);
    vp.Width  = (DWORD)width;
    vp.Height = (DWORD)height;
    vp.MinZ   = 0.0f;
    vp.MaxZ   = 1.0f;
    mDevice->SetViewport(&vp);
}

void GfxRenderingAPIDX9FF::SetScissor(int x, int y, int width, int height) {
    if (!mDevice) return;
    mDevice->SetRenderState(D3DRS_SCISSORTESTENABLE, TRUE);
    RECT rc;
    rc.left   = x;
    rc.right  = x + width;
    rc.bottom = mRenderTargetHeight - y;
    rc.top    = mRenderTargetHeight - y - height;
    mDevice->SetScissorRect(&rc);
}

void GfxRenderingAPIDX9FF::SetUseAlpha(bool useAlpha) {
    if (!mDevice || mAlphaBlend == useAlpha) return;
    mAlphaBlend = useAlpha;
    mDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, useAlpha ? TRUE : FALSE);
}

// ---------------------------------------------------------------------------
// DrawTriangles — core render call
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::ApplyTSSConfig(const TSSConfig& tss) {
    if (!mDevice) return;

    mDevice->SetTextureStageState(0, D3DTSS_COLOROP,  tss.colorOp0);
    mDevice->SetTextureStageState(0, D3DTSS_COLORARG1, tss.colorArg1_0);
    mDevice->SetTextureStageState(0, D3DTSS_COLORARG2, tss.colorArg2_0);
    mDevice->SetTextureStageState(0, D3DTSS_ALPHAOP,  tss.alphaOp0);
    mDevice->SetTextureStageState(0, D3DTSS_ALPHAARG1, tss.alphaArg1_0);
    mDevice->SetTextureStageState(0, D3DTSS_ALPHAARG2, tss.alphaArg2_0);

    if (tss.useStage1) {
        mDevice->SetTextureStageState(1, D3DTSS_COLOROP,  tss.colorOp1);
        mDevice->SetTextureStageState(1, D3DTSS_COLORARG1, tss.colorArg1_1);
        mDevice->SetTextureStageState(1, D3DTSS_COLORARG2, tss.colorArg2_1);
        mDevice->SetTextureStageState(1, D3DTSS_ALPHAOP,  tss.alphaOp1);
        mDevice->SetTextureStageState(1, D3DTSS_ALPHAARG1, tss.alphaArg1_1);
        mDevice->SetTextureStageState(1, D3DTSS_ALPHAARG2, tss.alphaArg2_1);
    } else {
        mDevice->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        mDevice->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
    }

    if (tss.setTextureFactor) {
        mDevice->SetRenderState(D3DRS_TEXTUREFACTOR, tss.textureFactor);
    }
}

void GfxRenderingAPIDX9FF::DrawTriangles(float bufVbo[], size_t bufVboLen, size_t bufVboNumTris) {
    if (!mDevice || !mVertexBuffer || bufVboNumTris == 0) return;

    // Bind textures
    for (int t = 0; t < 2; t++) {
        uint32_t id = mCurrentTextureIds[t];
        IDirect3DTexture9* tex = (id < mTextures.size()) ? mTextures[id].texture : nullptr;
        mDevice->SetTexture(t, tex);
    }

    // Apply shader (texture stage states)
    if (mCurrentProgram) {
        ApplyTSSConfig(mCurrentProgram->tss);
    }

    // Convert the float VBO (wx wy wz nx ny nz r g b a u v per vertex) to VertexD3D9FF structs.
    size_t numVerts = bufVboNumTris * 3;

    void* lockPtr = nullptr;
    UINT lockBytes = (UINT)(numVerts * sizeof(VertexD3D9FF));
    HRESULT hr = mVertexBuffer->Lock(0, lockBytes, &lockPtr, D3DLOCK_DISCARD);
    if (FAILED(hr)) {
        SPDLOG_WARN("[DX9FF] VB Lock failed: 0x{:08X}", (unsigned)hr);
        return;
    }

    auto* verts = static_cast<VertexD3D9FF*>(lockPtr);
    for (size_t i = 0; i < numVerts; i++) {
        const float* src = bufVbo + i * D3D9FF_VBO_FLOATS_PER_VERTEX;
        verts[i].x       = src[0];
        verts[i].y       = src[1];
        verts[i].z       = src[2];
        verts[i].nx      = src[3];
        verts[i].ny      = src[4];
        verts[i].nz      = src[5];
        verts[i].diffuse = FloatRGBAtoD3DCOLOR(src[6], src[7], src[8], src[9]);
        verts[i].u       = src[10];
        verts[i].v       = src[11];
    }
    mVertexBuffer->Unlock();

    mDevice->SetFVF(D3D9FF_FVF);
    mDevice->SetStreamSource(0, mVertexBuffer, 0, sizeof(VertexD3D9FF));
    mDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, (UINT)bufVboNumTris);
}

// ---------------------------------------------------------------------------
// Framebuffer operations
// ---------------------------------------------------------------------------
int GfxRenderingAPIDX9FF::CreateFramebuffer() {
    mFrameBuffers.push_back({});
    int id = (int)mFrameBuffers.size() - 1;

    // Every framebuffer gets its own texture slot so the GUI can use it as an ImGui texture.
    mFrameBuffers[id].textureId = NewTexture();
    return id;
}

void GfxRenderingAPIDX9FF::UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height,
                                                        uint32_t /*msaaLevel*/, bool /*openglInvertY*/,
                                                        bool renderTarget, bool hasDepthBuffer,
                                                        bool /*canExtractDepth*/) {
    if (!mDevice || fbId < 0 || fbId >= (int)mFrameBuffers.size()) return;
    auto& fb = mFrameBuffers[fbId];

    if (fb.width == width && fb.height == height) return;
    fb.width  = width;
    fb.height = height;

    if (fb.colorSurface) { fb.colorSurface->Release(); fb.colorSurface = nullptr; }
    if (fb.depthSurface) { fb.depthSurface->Release(); fb.depthSurface = nullptr; }

    if (fbId == 0) {
        // fb 0 is the swap-chain back-buffer — resolved at draw time via GetBackBuffer.
        return;
    }

    if (renderTarget) {
        uint32_t id = fb.textureId;
        if (id < mTextures.size() && mTextures[id].texture) {
            mTextures[id].texture->Release();
            mTextures[id].texture = nullptr;
        }
        IDirect3DTexture9* rt = nullptr;
        mDevice->CreateTexture(width, height, 1, D3DUSAGE_RENDERTARGET,
                               D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &rt, nullptr);
        if (id < mTextures.size()) mTextures[id].texture = rt;
        if (rt) rt->GetSurfaceLevel(0, &fb.colorSurface);
    }

    if (hasDepthBuffer) {
        mDevice->CreateDepthStencilSurface(width, height, D3DFMT_D24S8,
                                           D3DMULTISAMPLE_NONE, 0, TRUE,
                                           &fb.depthSurface, nullptr);
        fb.hasDepthBuffer = true;
    }
}

void GfxRenderingAPIDX9FF::StartDrawToFramebuffer(int fbId, float /*noiseScale*/) {
    if (!mDevice || fbId < 0 || fbId >= (int)mFrameBuffers.size()) return;
    mCurrentFramebuffer = fbId;

    IDirect3DSurface9* color  = nullptr;
    IDirect3DSurface9* depth  = nullptr;

    if (fbId == 0) {
        mDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &color);
        // Use the auto depth stencil created during Init.
        mDevice->GetDepthStencilSurface(&depth);
    } else {
        color = mFrameBuffers[fbId].colorSurface;
        depth = mFrameBuffers[fbId].depthSurface;
    }

    if (color) mDevice->SetRenderTarget(0, color);
    if (depth) mDevice->SetDepthStencilSurface(depth);

    if (fbId == 0) {
        // Release the local reference obtained via GetBackBuffer / GetDepthStencilSurface.
        if (color) color->Release();
        if (depth) depth->Release();
    }

    if (fbId < (int)mFrameBuffers.size()) {
        mRenderTargetHeight = (int32_t)mFrameBuffers[fbId].height;
        if (mRenderTargetHeight == 0) {
            RECT rc;
            GetClientRect(mWindowBackend->GetWindowHandle(), &rc);
            mRenderTargetHeight = rc.bottom - rc.top;
        }
    }
}

void GfxRenderingAPIDX9FF::ClearFramebuffer(bool color, bool depth) {
    if (!mDevice) return;
    DWORD flags = (color ? D3DCLEAR_TARGET : 0) | (depth ? D3DCLEAR_ZBUFFER : 0);
    if (flags) mDevice->Clear(0, nullptr, flags, D3DCOLOR_RGBA(0, 0, 0, 255), 1.0f, 0);
}

void GfxRenderingAPIDX9FF::CopyFramebuffer(int fbDstId, int fbSrcId,
                                            int srcX0, int srcY0, int srcX1, int srcY1,
                                            int dstX0, int dstY0, int dstX1, int dstY1) {
    if (!mDevice) return;
    if (fbSrcId < 0 || fbSrcId >= (int)mFrameBuffers.size()) return;
    if (fbDstId < 0 || fbDstId >= (int)mFrameBuffers.size()) return;

    IDirect3DSurface9* src = mFrameBuffers[fbSrcId].colorSurface;
    IDirect3DSurface9* dst = mFrameBuffers[fbDstId].colorSurface;
    if (!src || !dst) return;

    RECT srcRect = { srcX0, srcY0, srcX1, srcY1 };
    RECT dstRect = { dstX0, dstY0, dstX1, dstY1 };
    mDevice->StretchRect(src, &srcRect, dst, &dstRect, D3DTEXF_NONE);
}

void GfxRenderingAPIDX9FF::ReadFramebufferToCPU(int /*fbId*/, uint32_t /*width*/, uint32_t /*height*/,
                                                  uint16_t* /*rgba16Buf*/) {
    // Not implemented for the initial RTX Remix integration.
}

void GfxRenderingAPIDX9FF::ResolveMSAAColorBuffer(int /*fbIdTarget*/, int /*fbIdSrc*/) {
    // D3D9 MSAA is per-swap-chain; not needed for the current integration.
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDX9FF::GetPixelDepth(int /*fbId*/, const std::set<std::pair<float, float>>& /*coordinates*/) {
    // Depth read-back not yet implemented.
    return {};
}

void* GfxRenderingAPIDX9FF::GetFramebufferTextureId(int fbId) {
    if (fbId < 0 || fbId >= (int)mFrameBuffers.size()) return nullptr;
    uint32_t id = mFrameBuffers[fbId].textureId;
    return (id < mTextures.size()) ? (void*)mTextures[id].texture : nullptr;
}

// ---------------------------------------------------------------------------
// Texture filter / sRGB
// ---------------------------------------------------------------------------
void GfxRenderingAPIDX9FF::SetTextureFilter(FilteringMode mode) { mFilterMode = mode; }
FilteringMode GfxRenderingAPIDX9FF::GetTextureFilter() { return mFilterMode; }
void GfxRenderingAPIDX9FF::SetSrgbMode() { /* no-op */ }

ImTextureID GfxRenderingAPIDX9FF::GetTextureById(int id) {
    if (id < 0 || id >= (int)mTextures.size()) return nullptr;
    return reinterpret_cast<ImTextureID>(mTextures[id].texture);
}

// ---------------------------------------------------------------------------
// Misc query methods
// ---------------------------------------------------------------------------
const char* GfxRenderingAPIDX9FF::GetName() { return "Direct3D 9 Fixed-Function (RTX Remix)"; }

int GfxRenderingAPIDX9FF::GetMaxTextureSize() {
    if (!mDevice) return 4096;
    D3DCAPS9 caps;
    mDevice->GetDeviceCaps(&caps);
    return (int)std::min(caps.MaxTextureWidth, caps.MaxTextureHeight);
}

GfxClipParameters GfxRenderingAPIDX9FF::GetClipParameters() {
    // D3D9 clip space: z in [0,1], Y is not inverted (we flip in SetViewport/SetScissor).
    return { true, false };
}

} // namespace Fast

#endif // ENABLE_DX9
