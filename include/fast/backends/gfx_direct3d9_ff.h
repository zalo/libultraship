#pragma once
#ifdef ENABLE_DX9

#include <d3d9.h>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "gfx_rendering_api.h"

namespace Fast {

// Lightweight texture-stage-state snapshot derived from a N64 color combiner key.
struct TSSConfig {
    DWORD colorOp0, colorArg1_0, colorArg2_0;
    DWORD alphaOp0, alphaArg1_0, alphaArg2_0;
    bool  useStage1;
    DWORD colorOp1, colorArg1_1, colorArg2_1;
    DWORD alphaOp1, alphaArg1_1, alphaArg2_1;
    bool  setTextureFactor;
    D3DCOLOR textureFactor;
};

// D3D backends don't inherit from ShaderProgram (which has GL-specific fields).
// Like GfxRenderingAPIDX11, we store our own struct and C-style cast to ShaderProgram*.
struct ShaderProgramD3D9FF {
    uint64_t  shaderId0;
    uint32_t  shaderId1;
    uint8_t   numInputs;
    bool      usedTextures[2];
    bool      optAlpha;
    TSSConfig tss;
};

struct TextureD3D9 {
    IDirect3DTexture9* texture = nullptr;
    uint32_t width = 0, height = 0;
    bool     linearFiltering = false;
    uint8_t  cms = 0, cmt = 0;
};

struct FramebufferD3D9 {
    IDirect3DSurface9* colorSurface = nullptr;
    IDirect3DSurface9* depthSurface = nullptr;
    uint32_t textureId = 0;
    bool     hasDepthBuffer = false;
    uint32_t width = 0, height = 0;
};

class GfxWindowBackendDXGI;

// D3D9 fixed-function rendering backend.
// Designed to be intercepted by RTX Remix (dxvk-remix) for path-traced rendering of Ship of Harkinian.
// The backend emits world-space geometry + D3D9 SetLight() calls so Remix can reconstruct the scene.
class GfxRenderingAPIDX9FF final : public GfxRenderingAPI {
  public:
    explicit GfxRenderingAPIDX9FF(GfxWindowBackendDXGI* windowBackend);
    ~GfxRenderingAPIDX9FF() override;

    // RTX Remix integration points
    bool  OwnsLighting() const override { return true; }
    bool  RTXLightingMode() const override;
    void  CommitLights(const RSP* rsp) override;
    void  CommitProjection(const float pMatrix[4][4]) override;

    // GfxRenderingAPI interface
    const char* GetName() override;
    int         GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;

    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint32_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint32_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;

    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linearFilter, uint32_t cms, uint32_t cmt) override;
    void DeleteTexture(uint32_t texId) override;
    void SelectTextureFb(int fbId) override;

    void SetDepthTestAndMask(bool depthTest, bool zUpd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;

    void DrawTriangles(float bufVbo[], size_t bufVboLen, size_t bufVboNumTris) override;

    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;

    int  CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height, uint32_t msaaLevel,
                                     bool openglInvertY, bool renderTarget, bool hasDepthBuffer,
                                     bool canExtractDepth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1,
                         int dstX0, int dstY0, int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) override;

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fbId, const std::set<std::pair<float, float>>& coordinates) override;

    void*         GetFramebufferTextureId(int fbId) override;
    void          SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void          SetSrgbMode() override;
    ImTextureID   GetTextureById(int id) override;

    IDirect3DDevice9* GetDevice() const { return mDevice; }

  private:
    void ApplyTSSConfig(const TSSConfig& tss);
    TSSConfig BuildTSSFromCC(uint64_t shaderId0, uint32_t shaderId1, bool& outAlpha, bool outUsedTex[2]);
    void ResetDeviceState();

    GfxWindowBackendDXGI* mWindowBackend = nullptr;

    HMODULE               mD3D9Module   = nullptr;
    IDirect3D9*           mD3D          = nullptr;
    IDirect3DDevice9*     mDevice       = nullptr;
    IDirect3DVertexBuffer9* mVertexBuffer = nullptr;

    std::vector<TextureD3D9>     mTextures;
    std::vector<FramebufferD3D9> mFrameBuffers;

    std::map<std::pair<uint64_t, uint32_t>, ShaderProgramD3D9FF> mShaderPool;
    ShaderProgramD3D9FF* mCurrentProgram = nullptr;

    uint32_t      mCurrentTextureIds[2] = {};
    int           mCurrentFramebuffer   = 0;
    int32_t       mRenderTargetHeight   = 480;
    int           mPrevLightCount       = 0;
    FilteringMode mFilterMode           = FILTER_THREE_POINT;
    bool          mAlphaBlend           = false;

    float mCachedPMatrix[4][4] = {};
    bool  mPMatrixDirty        = true;
};

} // namespace Fast

#endif // ENABLE_DX9
