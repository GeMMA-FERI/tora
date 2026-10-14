@echo off
setlocal
cd /d "%~dp0.."
if not defined LibTorchDir set "LibTorchDir=C:\GitHub\libtorch\libtorch"
if not defined GemmaFusionSdkDir set "GemmaFusionSdkDir=%LOCALAPPDATA%\GFSDK"
if not defined VCPKG_ROOT exit /b 1
set "ToraVcpkgRoot=%VCPKG_ROOT%"
if not defined VSCMD_VER (
  call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)
if not exist _build\tensor-region-test mkdir _build\tensor-region-test
cl /nologo /std:c++20 /EHsc /MD /utf-8 /Iextensions\GF.Extensions.PyTorch /I"%LibTorchDir%\include" /I"%LibTorchDir%\include\torch\csrc\api\include" /I"%GemmaFusionSdkDir%\Include" /I"%ToraVcpkgRoot%\installed\x64-windows-static-md\include" tests\cpp\tensor_region_abi.cpp /Fo_build\tensor-region-test\tensor_region_abi.obj /Fe_build\tensor-region-test\tensor_region_abi.exe /link /LIBPATH:"%LibTorchDir%\lib" /LIBPATH:"%GemmaFusionSdkDir%\Lib\x64" torch_cpu.lib c10.lib FrameworkData.lib
if errorlevel 1 exit /b 1
set "PATH=%LibTorchDir%\lib;%LibTorchDir%\bin;%GemmaFusionSdkDir%\Lib\x64;%PATH%"
_build\tensor-region-test\tensor_region_abi.exe
exit /b %ERRORLEVEL%
