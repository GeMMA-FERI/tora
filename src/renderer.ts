import { makeDisplayName, makeNodeClassName, signatureUuid, uuidToCppCtor } from "./naming.js";
import { renderInputAliasType, renderReturnAliasType } from "./templates.js";
import type { TorchFunction } from "./types.js";

function safeAliasName(input: string): string {
  return input.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^([0-9])/, "_$1");
}

export function renderNodeDefinition(fn: TorchFunction): string {
  const className = makeNodeClassName(fn);
  const displayName = makeDisplayName(fn);
  const cppFunctionName = fn.overloadName && fn.args.some((arg) => arg.rawType.includes("!"))
    ? `${fn.baseName}_outf`
    : fn.baseName;
  const uuid = signatureUuid(fn.schemaSignature);

  const inputAliases = fn.args
    .map((arg) => `    using In_${safeAliasName(arg.name)} = ${renderInputAliasType(arg)};`)
    .join("\n");

  const outputAliases = fn.returns
    .map((ret, i) => `    using Out_${safeAliasName(ret.name ?? `output_${i}`)} = ${renderReturnAliasType(ret, i)};`)
    .join("\n");

  const registerInputs = fn.args
    .map((arg) =>
      arg.optional
        ? `        RegisterOptionalInput<In_${safeAliasName(arg.name)}>();`
        : `        RegisterInput<In_${safeAliasName(arg.name)}>();`
    )
    .join("\n");

  const registerOutputs = fn.returns
    .map((ret, i) => `        RegisterOutput<Out_${safeAliasName(ret.name ?? `output_${i}`)}>();`)
    .join("\n");

  const fetchInputs = fn.args
    .map((arg) => {
      const n = safeAliasName(arg.name);
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

  return `struct ${className} : GF::Extensions::impl::DataProcessorBridgeImpl<${className}> {
    static constexpr auto name() { return std::string_view("${displayName}"); }
    static constexpr auto description() { return std::string_view("Auto-generated from PyTorch signature: ${fn.rawFunc}"); }
    static constexpr auto interface_id() { return Core::interfaces::interface_id_storage<GF::Interfaces::Ext::IDataProcessor>::value; }
    // {${uuid.toUpperCase()}}
    static constexpr auto implementation_id() { return ${uuidToCppCtor(uuid)}; }
    static auto create_instance() { return Core::make_base_obj<${className}>().as<Core::Object>(); }

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
