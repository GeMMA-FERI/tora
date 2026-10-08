import type { ArgKind, TorchArgument, TorchFunction, TorchReturn } from "./types.js";

function splitTopLevel(input: string, separator: string): string[] {
  const parts: string[] = [];
  let current = "";
  let paren = 0;
  let bracket = 0;
  let quote: string | null = null;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (quote) {
      current += ch;
      if (ch === quote && input[i - 1] !== "\\") {
        quote = null;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      current += ch;
      continue;
    }

    if (ch === "(") paren++;
    if (ch === ")") paren--;
    if (ch === "[") bracket++;
    if (ch === "]") bracket--;

    if (ch === separator && paren === 0 && bracket === 0) {
      const token = current.trim();
      if (token.length > 0) parts.push(token);
      current = "";
      continue;
    }

    current += ch;
  }

  const tail = current.trim();
  if (tail.length > 0) parts.push(tail);
  return parts;
}

function splitOnceTopLevel(input: string, separator: string): [string, string | undefined] {
  let paren = 0;
  let bracket = 0;
  let quote: string | null = null;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (quote) {
      if (ch === quote && input[i - 1] !== "\\") {
        quote = null;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (ch === "(") paren++;
    if (ch === ")") paren--;
    if (ch === "[") bracket++;
    if (ch === "]") bracket--;

    if (ch === separator && paren === 0 && bracket === 0) {
      return [input.slice(0, i).trim(), input.slice(i + 1).trim()];
    }
  }

  return [input.trim(), undefined];
}

function classifyKind(rawType: string): ArgKind {
  const normalized = rawType.replace(/\(.*?\)/g, "").trim();

  if (/^Tensor(\?|\[\])*/.test(normalized)) return "Tensor";
  if (normalized.startsWith("ScalarType")) return "ScalarType";
  if (normalized.startsWith("Scalar")) return "Scalar";
  if (normalized.startsWith("SymInt")) return "SymInt";
  if (normalized.startsWith("int")) return "Int";
  if (normalized.startsWith("float")) return "Float";
  if (normalized.startsWith("bool")) return "Bool";
  if (normalized.startsWith("str")) return "Str";
  if (normalized.startsWith("Layout")) return "Layout";
  if (normalized.startsWith("Device")) return "Device";
  if (normalized.startsWith("MemoryFormat")) return "MemoryFormat";
  if (normalized.startsWith("Generator")) return "Generator";
  if (normalized.startsWith("Dimname")) return "Dimname";
  if (normalized.startsWith("Stream")) return "Stream";
  if (normalized.startsWith("QScheme")) return "QScheme";

  return "Unknown";
}

function parseTypeMeta(rawType: string): { optional: boolean; list: boolean } {
  return {
    optional: rawType.includes("?"),
    list: rawType.includes("[]") || /\[[0-9]*\]/.test(rawType)
  };
}

function parseArgument(token: string, kwOnly: boolean): TorchArgument {
  const [left, defaultValue] = splitOnceTopLevel(token, "=");
  const lastSpace = left.lastIndexOf(" ");
  if (lastSpace < 0) {
    throw new Error(`Invalid argument token: ${token}`);
  }

  const rawType = left.slice(0, lastSpace).trim();
  const name = left.slice(lastSpace + 1).trim();
  const meta = parseTypeMeta(rawType);

  return {
    name,
    rawType,
    kind: classifyKind(rawType),
    optional: meta.optional,
    list: meta.list,
    listLength: /\[(\d+)\]/.test(rawType) ? Number(/\[(\d+)\]/.exec(rawType)![1]) : undefined,
    kwOnly,
    defaultValue
  };
}

function parseReturnToken(token: string): TorchReturn {
  const trimmed = token.trim();
  const lastSpace = trimmed.lastIndexOf(" ");
  const hasName = lastSpace > -1 && !trimmed.slice(lastSpace + 1).includes(")");

  const rawType = hasName ? trimmed.slice(0, lastSpace).trim() : trimmed;
  const name = hasName ? trimmed.slice(lastSpace + 1).trim() : undefined;
  const meta = parseTypeMeta(rawType);

  return {
    name,
    rawType,
    kind: classifyKind(rawType),
    optional: meta.optional,
    list: meta.list
  };
}

function parseReturns(returnsPart: string): TorchReturn[] {
  const trimmed = returnsPart.trim();
  if (trimmed.startsWith("(") && trimmed.endsWith(")")) {
    const inner = trimmed.slice(1, -1).trim();
    if (inner.length === 0) return [];
    return splitTopLevel(inner, ",").map(parseReturnToken);
  }
  return [parseReturnToken(trimmed)];
}

export function parseFuncSchema(func: string, tags: string[], variants: string[]): TorchFunction {
  const arrowIndex = func.lastIndexOf("->");
  if (arrowIndex < 0) {
    throw new Error(`Invalid func schema (missing ->): ${func}`);
  }

  const left = func.slice(0, arrowIndex).trim();
  const returnsPart = func.slice(arrowIndex + 2).trim();

  const openParen = left.indexOf("(");
  const closeParen = left.lastIndexOf(")");
  if (openParen < 0 || closeParen < openParen) {
    throw new Error(`Invalid func schema (bad args list): ${func}`);
  }

  const opName = left.slice(0, openParen).trim();
  const argsBlock = left.slice(openParen + 1, closeParen);

  const [baseName, overloadName] = splitOnceTopLevel(opName, ".");
  const argTokens = splitTopLevel(argsBlock, ",");
  const args: TorchArgument[] = [];
  let kwOnly = false;

  for (const token of argTokens) {
    if (token === "*") {
      kwOnly = true;
      continue;
    }
    args.push(parseArgument(token, kwOnly));
  }

  const returns = parseReturns(returnsPart);
  const schemaSignature = `${opName}(${args.map((a) => `${a.rawType} ${a.name}`).join(",")}) -> ${returnsPart}`;

  return {
    rawFunc: func,
    baseName,
    overloadName,
    schemaSignature,
    args,
    returns,
    tags,
    variants
  };
}
