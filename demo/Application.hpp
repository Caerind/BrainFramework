#pragma once

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>
#include <d3d11.h>
#include <tchar.h>
#include <functional>

class Application
{
public:
    Application();
    ~Application();

    void Run(std::function<void()>&& loopFunction);

private:
    void Init();
    void Cleanup();

    // Forward declarations of helper functions
    bool CreateDeviceD3D(HWND hWnd);
    void CleanupDeviceD3D();
    void CreateRenderTarget();
    void CleanupRenderTarget();

    static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    WNDCLASSEXW mWc;
    HWND mHwnd;
};
