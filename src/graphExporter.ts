import { randomUUID } from "node:crypto";
import path from "node:path";
import { bindProgram, type ValueSource } from "./binder.js";
import { EDITOR_PROFILE, genericIO, type EditorNodeBase } from "./editor.js";
import type { Program } from "./program.js";
import type { NodeRegistry } from "./registry.js";

export interface GraphNode {
  NodeBase: string;
  InstanceUuid: string;
  Metadata: object;
  Properties: Record<string, number | boolean | string>;
  Bounds: [number, number, number, number];
}
export interface NodeGraph {
  Nodes: Record<string, GraphNode>;
  Edges: Record<string, { U: string; V: string; UT: string; VT: string }>;
  NodeBases: Record<string, EditorNodeBase>;
  Tora?: { capture: NonNullable<Program["capture"]> };
}

function absoluteResourcePath(file: string): boolean {
  // Windows root-relative paths (\file) still depend on the current drive.
  return path.isAbsolute(file) && !(process.platform === "win32" && /^[\\/]$/.test(path.parse(file).root));
}

export function exportNodeGraph(program: Program, registry: NodeRegistry,
  options: { resourceDirectory?: string } = {}): NodeGraph {
  if (registry.editorProfile !== EDITOR_PROFILE) {
    throw new Error(`Expected editor profile ${EDITOR_PROFILE}; regenerate the registry with this Tora version`);
  }
  const bound = bindProgram(program, registry);
  if (bound.resources.length && !registry.tensorLoader) throw new Error("Tensor resources require loader metadata; regenerate the registry");
  const graph: NodeGraph = { Nodes: {}, Edges: {}, NodeBases: {} };
  // Advisory provenance only: the editor currently does not enforce these
  // capture assumptions and may omit this field when it exports the graph.
  if (program.capture) graph.Tora = { capture: structuredClone(program.capture) };
  const nodeIds = new Map<string, string>();
  const depths = new Map<string, number>();
  const rows = new Map<number, number>();
  let edgeCount = 0;
  function addNode(implementationId: string, base: EditorNodeBase, metadata: object, depth: number,
    properties: GraphNode["Properties"] = {}): string {
    const id = String(Object.keys(graph.Nodes).length);
    const x = depth * 320;
    const y = rows.get(depth) ?? 0;
    const height = 90 + 35 * Math.max(Object.keys(base.Inputs).length, Object.keys(base.Outputs).length);
    rows.set(depth, y + height + 60);
    graph.Nodes[id] = { NodeBase: implementationId, InstanceUuid: randomUUID(), Metadata: metadata,
      Properties: properties, Bounds: [x, y, x + 240, y + height] };
    const previous = graph.NodeBases[implementationId];
    if (previous && JSON.stringify(previous) !== JSON.stringify(base)) {
      throw new Error(`Conflicting node definitions for implementation ${implementationId}`);
    }
    graph.NodeBases[implementationId] = base;
    return id;
  }
  let inputId: string | undefined;
  if (bound.inputs.length) {
    const io = genericIO("inputs", bound.inputs.map(input => ({ key: input.id, type: input.type })));
    inputId = addNode(io.implementationId, io.nodeBase, io.metadata, 0);
  }
  const resourceNodes = new Map<string, string>();
  if (bound.resources.length) {
    const binding = registry.tensorLoader!;
    const snapshot = (ports: typeof binding.inputs | typeof binding.outputs) => Object.fromEntries(
      Object.values(ports).map(port => [port.key, { Description: port.description, ...port.editorType }]));
    const base: EditorNodeBase = { Name: binding.name, Description: binding.description,
      Inputs: snapshot(binding.inputs), Outputs: snapshot(binding.outputs) };
    for (const resource of bound.resources) {
      const directory = options.resourceDirectory;
      const absolute = absoluteResourcePath(resource.file);
      if (!absolute && path.parse(resource.file).root) {
        throw new Error(`Tensor resource ${resource.id}: file must be absolute or relative without a drive/root prefix`);
      }
      if (!absolute && (!directory || !absoluteResourcePath(directory))) {
        throw new Error(`Tensor resource ${resource.id}: relative file requires an absolute resourceDirectory`);
      }
      const properties: GraphNode["Properties"] = {
        [binding.inputs.file.key]: resource.file,
        [binding.inputs.name.key]: resource.key,
      };
      if (!absolute) properties[binding.inputs.base.key] = directory!;
      resourceNodes.set(resource.id, addNode(binding.implementationId, base, {}, 0, properties));
    }
  }
  function sourceEndpoint(source: ValueSource): { node: string; key: string; depth: number } {
    if ("input" in source) {
      if (inputId === undefined) throw new Error(`Missing graph input ${source.input}`);
      return { node: inputId, key: source.input, depth: 0 };
    }
    if ("resource" in source) {
      const node = resourceNodes.get(source.resource);
      if (node === undefined) throw new Error(`Missing tensor resource ${source.resource}`);
      return { node, key: registry.tensorLoader!.outputs.tensor.key, depth: 0 };
    }
    const node = nodeIds.get(source.operation);
    if (node === undefined) throw new Error(`Missing producer ${source.operation}`);
    return { node, key: source.key, depth: depths.get(source.operation)! };
  }
  function connect(source: ValueSource, destination: string, key: string) {
    const endpoint = sourceEndpoint(source);
    graph.Edges[String(edgeCount++)] = { U: endpoint.node, V: destination, UT: endpoint.key, VT: key };
  }
  const bindings = new Map(registry.operations.map(binding => [binding.operator, binding]));
  for (const operation of bound.operations) {
    const binding = bindings.get(operation.operator)!;
    const properties: [string, number | boolean | string][] = [];
    let depth = 1;
    for (const input of operation.inputs) {
      if ("literal" in input.source) {
        // The editor represents absent optional values by omitting the property.
        if (input.source.literal !== null) properties.push([input.key, input.source.literal]);
      } else {
        depth = Math.max(depth, sourceEndpoint(input.source).depth + 1);
      }
    }
    const base: EditorNodeBase = {
      Name: binding.name, Description: binding.description,
      Inputs: Object.fromEntries(binding.inputs.map(port => [port.key, { Description: "", ...port.editorType }])),
      Outputs: Object.fromEntries(binding.outputs.map(port => [port.key, { Description: "", ...port.editorType }])),
    };
    const node = addNode(operation.implementationId, base, {}, depth, Object.fromEntries(properties));
    nodeIds.set(operation.id, node);
    depths.set(operation.id, depth);
    for (const input of operation.inputs) {
      if (!("literal" in input.source)) connect(input.source, node, input.key);
    }
  }
  if (bound.outputs.length) {
    const io = genericIO("outputs", bound.outputs.map(output => ({ key: output.name, type: output.source.type })));
    const depth = 1 + Math.max(...bound.outputs.map(output => sourceEndpoint(output.source).depth));
    const node = addNode(io.implementationId, io.nodeBase, io.metadata, depth);
    for (const output of bound.outputs) connect(output.source, node, output.name);
  }
  return graph;
}
