import { makeDisplayName, makeNodeClassName, signatureUuid, uuidToCppCtor } from "./naming.js";
import { renderReturnAliasType } from "./templates.js";
import type { TorchFunction } from "./types.js";
import { resolveInputPorts, safeAliasName, integerDefault } from "./inputPorts.js";

export function renderNodeDefinition(fn: TorchFunction): string {
  const className = makeNodeClassName(fn);
  const displayName = fn.displayName ?? makeDisplayName(fn);
  const cppFunctionName = fn.overloadName && fn.args.some((arg) => arg.rawType.includes("!"))
    ? `${fn.baseName}_outf`
    : fn.baseName;
  const uuid = signatureUuid(fn.schemaSignature);

  const ports = resolveInputPorts(fn);
  const inputAliases = ports
    .map(port => `    using In_${port.alias} = ${port.cppType};`)
    .join("\n");

  const outputAliases = fn.returns
    .map((ret, i) => `    using Out_${safeAliasName(ret.name ?? `output_${i}`)} = ${renderReturnAliasType(ret, i)};`)
    .join("\n");

  const registerInputs = ports
    .map(port => {
      if (port.argument.optional) return `        RegisterOptionalInput<In_${port.alias}>();`;
      const value = fn.arguments && ["Int", "SymInt"].includes(port.argument.kind)
        ? integerDefault(port.argument, port.split?.index) : undefined;
      const expression = value === undefined ? "" : `::Data::Variant(static_cast<Core::Int64>(${value}))`;
      return `        RegisterInput<In_${port.alias}>(${expression});`;
    })
    .join("\n");

  const registerOutputs = fn.returns
    .map((ret, i) => `        RegisterOutput<Out_${safeAliasName(ret.name ?? `output_${i}`)}>();`)
    .join("\n");

  const fetchInputs = fn.args
    .map((arg) => {
      const n = safeAliasName(arg.name);
      const split = ports.filter(port => port.argument === arg && port.split);
      if (split.length) return `        const std::array<int64_t, ${split.length}> torch_${n}{ ${split.map(port => `static_cast<int64_t>(inputs.Get<In_${port.alias}>())`).join(", ")} };`;
      if (arg.optional) {
        return [
          `        std::optional<decltype(PyTorch::ToTorchValue(inputs.Get<In_${n}>()))> torch_${n};`,
          `        if (inputs.ContainsValue<In_${n}>()) {`,
          `            torch_${n} = PyTorch::ToTorchValue(inputs.Get<In_${n}>());`,
          "        }"
        ].join("\n");
      }
      return `        auto torch_${n} = PyTorch::ToTorchValue(inputs.Get<In_${n}>());`;
    })
    .join("\n");

  const callArgs = fn.args.map((arg) => `torch_${safeAliasName(arg.name)}`).join(", ");

  const setOutputs =
    fn.returns.length <= 1
      ? `        outputs.Set<Out_${safeAliasName(fn.returns[0]?.name ?? "output_0")}>((PyTorch::FromTorchValue(result)));`
      : fn.returns
          .map((ret, i) => {
            const n = safeAliasName(ret.name ?? `output_${i}`);
            return `        outputs.Set<Out_${n}>(PyTorch::FromTorchValue(std::get<${i}>(result)));`;
          })
          .join("\n");

  return `${ports.some(port => port.split) ? "#include <array>\n\n" : ""}struct ${className} : GF::Extensions::impl::DataProcessorBridgeImpl<${className}> {
    static constexpr auto name() { return std::string_view(${JSON.stringify(displayName)}); }
    static constexpr auto description() { return std::string_view(${JSON.stringify(fn.description ?? `Auto-generated from PyTorch signature: ${fn.rawFunc}`)}); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }
    // {${uuid.toUpperCase()}}
    static constexpr auto implementation_id() { return ${uuidToCppCtor(uuid)}; }
    static auto create_instance() { return Core::make_base_obj<${className}>().as<Core::Object>(); }

    static ::Data::Properties create_metadata()
    {
        ::Data::Properties metadata;
        AddNamespace<${JSON.stringify(fn.namespace ?? "PyTorch::Uncategorized")}>(metadata);
        return metadata;
    }

${inputAliases}
${outputAliases}

    ${className}()
    {
${registerInputs}
${registerOutputs}
    }

    void Process(const Data::Properties& inputs, Data::Properties outputs)
    {
${fetchInputs}

        auto result = at::${cppFunctionName}(${callArgs});
${setOutputs}
    }
};
`;
}
