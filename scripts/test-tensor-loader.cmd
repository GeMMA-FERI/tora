@echo off
setlocal
cd /d "%~dp0.."
if not defined LibTorchDir set "LibTorchDir=C:\GitHub\libtorch\libtorch"
if not defined VCPKG_ROOT exit /b 1
set "ToraVcpkgRoot=%VCPKG_ROOT%"
if not defined VSCMD_VER (
  call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)
if not exist _build\loader-test mkdir _build\loader-test
cl /nologo /std:c++20 /EHsc /MD /utf-8 /Iextensions\GF.Extensions.PyTorch /I"%LibTorchDir%\include" /I"%ToraVcpkgRoot%\installed\x64-windows-static-md\include" tests\cpp\load_tensor.cpp /Fo_build\loader-test\load_tensor.obj /Fe_build\loader-test\load_tensor.exe /link /LIBPATH:"%LibTorchDir%\lib" torch_cpu.lib c10.lib
if errorlevel 1 exit /b 1
set "PATH=%LibTorchDir%\lib;%LibTorchDir%\bin;%PATH%"
_build\capture-venv\Scripts\python.exe tests\python\check_tensor_loader.py
exit /b %ERRORLEVEL%
