import { createHash } from "node:crypto";
import type { TorchFunction } from "./types.js";

export function toPascalCase(input: string): string {
  return input
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");
}

export function makeNodeClassName(fn: TorchFunction): string {
  const overload = fn.overloadName ? `_${fn.overloadName}` : "";
  const suffix = signatureUuid(fn.schemaSignature).replaceAll("-", "");
  return `${toPascalCase(fn.baseName + overload)}Processor_${suffix}`;
}

export function makeDisplayName(fn: TorchFunction): string {
  const raw = fn.overloadName ? `${fn.baseName} ${fn.overloadName}` : fn.baseName;
  return raw
    .replace(/[._]+/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

export function signatureUuid(signature: string): string {
  const hash = createHash("sha1").update(signature).digest();
  const bytes = Buffer.from(hash.subarray(0, 16));
  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = bytes.toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}

export function uuidToCppCtor(uuid: string): string {
  const [a, b, c, d, e] = uuid.split("-");
  const tail = `${d}${e}`.match(/../g) ?? [];
  const bytes = tail.map((x) => `0x${x}`).join(", ");
  return `Core::Uuid(0x${a}, 0x${b}, 0x${c}, { ${bytes} })`;
}
