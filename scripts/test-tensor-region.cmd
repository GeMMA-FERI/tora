@echo off
setlocal
cd /d "%~dp0.."
if not defined LibTorchDir set "LibTorchDir=C:\GitHub\libtorch\libtorch"
if not defined VSCMD_VER (
  call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)
if not exist _build\tensor-region-test mkdir _build\tensor-region-test
cl /nologo /std:c++20 /EHsc /MD /utf-8 /Iextensions\GF.Extensions.PyTorch /I"%LibTorchDir%\include" tests\cpp\tensor_region.cpp /Fo_build\tensor-region-test\tensor_region.obj /Fe_build\tensor-region-test\tensor_region.exe /link /LIBPATH:"%LibTorchDir%\lib" torch_cpu.lib c10.lib
if errorlevel 1 exit /b 1
set "PATH=%LibTorchDir%\lib;%LibTorchDir%\bin;%PATH%"
_build\tensor-region-test\tensor_region.exe
exit /b %ERRORLEVEL%
