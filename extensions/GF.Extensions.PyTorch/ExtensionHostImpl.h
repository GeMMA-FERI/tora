#pragma once
#include <GF/Extensions/Bridges/ExtensionHostBridgeImpl.hxx>
#include "generated/GeneratedProcessorsManifest.hpp"
#include "TensorDiagnosticProcessors.hpp"
#include "TensorFromTypedArrayProcessor.hpp"
#include "LoadTensorProcessor.hpp"

namespace ExtensionHost::impl {

	struct ExtensionHostImpl : GF::Extensions::impl::ExtensionHostImplBase {
		ExtensionHostImpl() : ExtensionHostImplBase()
		{
			Add<TensorFromTextProcessor>();
			Add<TensorToTextProcessor>();
			Add<TensorFromTypedArrayProcessor>();
			Add<LoadTensorProcessor>();
			GFExtensionsPyTorchGenerated::ForEachProcessor([this]<typename TProcessor>() {
				this->template Add<TProcessor>();
			});
		}
	};
}
