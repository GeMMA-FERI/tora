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
call npm.cmd run build
if errorlevel 1 exit /b 1
node scripts\prepare-convolution-test.mjs
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /Zc:__cplusplus /EHsc /MD /bigobj /utf-8 /wd4251 /wd4275 /DXAPIAN_STATIC /Iextensions\GF.Extensions.PyTorch /I_build\convolution-test /I"%GemmaFusionSdkDir%\Include" /I"%LibTorchDir%\include" /I"%LibTorchDir%\include\torch\csrc\api\include" /I"%ToraVcpkgRoot%\installed\x64-windows-static-md\include" tests\cpp\convolution.cpp /Fo_build\convolution-test\convolution.obj /Fe_build\convolution-test\convolution.exe /link /LIBPATH:"%LibTorchDir%\lib" /LIBPATH:"%GemmaFusionSdkDir%\Lib\x64" torch_cpu.lib c10.lib FrameworkData.lib Platform.lib GF.Core.lib
if errorlevel 1 exit /b 1
set "PATH=%LibTorchDir%\lib;%LibTorchDir%\bin;%GemmaFusionSdkDir%\Lib\x64;%PATH%"
_build\convolution-test\convolution.exe
exit /b %ERRORLEVEL%
