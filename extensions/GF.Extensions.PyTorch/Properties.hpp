#pragma once
#include <GF/Extensions/Bridges/DataProcessorBridgeImpl.hxx>
#include <Data/Property.hpp>
#include <Data/Tensor.hpp>
#include <Data/TypedArray.hpp>
#include <GF/Workspace/Workspace.hpp>

namespace Data {
	template <Core::StringLiteral PropertyName, Core::StringLiteral PropertyDescription>
	using TableProperty = Core::Property<PropertyName, PropertyDescription, Data::Table>;

	template <Core::StringLiteral PropertyName, Core::StringLiteral PropertyDescription>
	using FilePathProperty = Core::Property<PropertyName, PropertyDescription, Data::String>;

	template <Core::StringLiteral PropertyName, Core::StringLiteral PropertyDescription>
	using WorkspaceManagerProperty = Core::Property<PropertyName, PropertyDescription, GF::Api::Workspace::WorkspaceManager>;

	template <Core::StringLiteral PropertyName, Core::StringLiteral PropertyDescription>
	using TensorProperty = Core::Property<PropertyName, PropertyDescription, Data::Tensor>;

	template <Core::StringLiteral PropertyName, Core::StringLiteral PropertyDescription>
	using TypedArrayProperty = Core::Property<PropertyName, PropertyDescription, Data::TypedArray>;
}
