import { useEffect, useRef, useState } from "react";
import dagre from "dagre"

interface Relation {
  source: string;
  relation: string;
  target: string;
}

interface GraphPaneProps {
  nodes?: string[];
  relations?: Relation[];
}

interface NodePosition {
  x: number;
  y: number;
}

interface GraphAPIResponse {
  nodes: string[];
  relations: Relation[];
}

export default function GraphPane({ nodes: propsNodes = [], relations: propsRelations = [] }: GraphPaneProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<string[]>(propsNodes);
  const [relations, setRelations] = useState<Relation[]>(propsRelations);
  const [nodePositions, setNodePositions] = useState<Map<string, NodePosition>>(new Map());
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [panning, setPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const onMouseDownNode = (node: string, e: React.MouseEvent<SVGCircleElement>) => {
    const pos = nodePositions.get(node);
    if (!pos) return;

    setDraggingNode(node);
    setDragOffset({
      x: e.clientX - pos.x,
      y: e.clientY - pos.y
    });
  };

  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (panning) {
      setOffset({
        x: e.clientX - panStart.current.x,
        y: e.clientY - panStart.current.y,
      });
      return;
    }
    if (!draggingNode) {return};

    setNodePositions((prev) => {
      const newMap = new Map(prev);
      newMap.set(draggingNode, {
        x: e.clientX - dragOffset.x,
        y: e.clientY - dragOffset.y
      });
      return newMap;
    });
  };

  const onMouseUp = () => {
    setDraggingNode(null);
  };

  const onWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();

    const scaleAmount = -e.deltaY * 0.001; // smooth zoom
    setZoom((prev) => {
      let z = prev + scaleAmount;
      if (z < 0.1) z = 0.1;
      if (z > 4) z = 4;
      return z;
    });
  };

  // Fetch graph data on component mount
  useEffect(() => {
    const fetchGraphData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch("http://localhost:8000/graph");
        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }
        const data: GraphAPIResponse = await response.json();
        setNodes(data.nodes || []);
        setRelations(data.relations || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch graph data");
        console.error("Error fetching graph data:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  // Calculate node positions using scattered random layout
  useEffect(() => {
    if (nodes.length === 0 || relations.length === 0) return;

    const g = new dagre.graphlib.Graph();
    g.setGraph({
      rankdir: "LR",      // left → right layout
      nodesep: 50,        // spacing between nodes
      ranksep: 100        // spacing between layers
    });
    g.setDefaultEdgeLabel(() => ({}));

    // Add nodes with approximate sizes
    nodes.forEach((node) => {
      g.setNode(node, { width: 120, height: 50 });
    });

    // Add edges
    relations.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    dagre.layout(g);

    const graphInfo = g.graph();
    const graphWidth = graphInfo.width || 0;
    const graphHeight = graphInfo.height || 0;

    const offsetX = canvasSize.width / 2 - graphWidth / 2;
    const offsetY = canvasSize.height / 2 - graphHeight / 2;

    const positions = new Map<string, NodePosition>();

    nodes.forEach((node) => {
      const pos = g.node(node);
      positions.set(node, { x: pos.x, y: pos.y});
    });

    setNodePositions(positions);
  }, [nodes, canvasSize]);

  // Update canvas size on mount and resize
  useEffect(() => {
    const updateCanvasSize = () => {
      if (svgRef.current?.parentElement) {
        const rect = svgRef.current.parentElement.getBoundingClientRect();
        setCanvasSize({ width: rect.width, height: rect.height });
      }
    };

    updateCanvasSize();
    const resizeObserver = new ResizeObserver(updateCanvasSize);
    if (svgRef.current?.parentElement) {
      resizeObserver.observe(svgRef.current.parentElement);
    }

    return () => resizeObserver.disconnect();
  }, []);

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-slate-200 dark:border-slate-800 px-6 py-4 bg-white dark:bg-slate-900">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Knowledge Graph</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          {nodes.length} nodes • {relations.length} relations
        </p>
      </div>

      {/* Graph Canvas */}
      <div className="flex-1 overflow-auto p-4 relative">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-slate-400 dark:text-slate-600 animate-spin"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
              </div>
              <p className="text-slate-600 dark:text-slate-400 font-medium mb-1">
                Loading graph data...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-red-600 dark:text-red-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4v2m0 4v2m0-14a9 9 0 11-18 0 9 9 0 0118 0zm0 0v-2m0 4v2m0 4v2"
                  />
                </svg>
              </div>
              <p className="text-slate-600 dark:text-slate-400 font-medium mb-1">
                Failed to load graph
              </p>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                {error}
              </p>
            </div>
          </div>
        ) : nodes.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-slate-400 dark:text-slate-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <p className="text-slate-600 dark:text-slate-400 font-medium mb-1">
                No graph data
              </p>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                No nodes or relations available
              </p>
            </div>
          </div>
        ) : (
          <svg
            ref={svgRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className="bg-white dark:bg-slate-800 rounded-lg shadow-sm"
            style={{ minWidth: "100%", minHeight: "100%" }}
            onWheel={onWheel}
            onMouseDown={(e) => {
              // IMPORTANT: Only pan when clicking background, not nodes
              if (e.target === svgRef.current) {
                setPanning(true);
                panStart.current = {
                  x: e.clientX - offset.x,
                  y: e.clientY - offset.y,
                };
              }
            }}
            onMouseMove={onMouseMove}
            onMouseUp={() => {
              setPanning(false);
              onMouseUp();
            }}
          >
            <g transform={`translate(${offset.x}, ${offset.y}) scale(${zoom})`}>
            {/* Render edges/relations */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="10"
                refX="9"
                refY="3"
                orient="auto"
              >
                <polygon
                  points="0 0, 10 3, 0 6"
                  fill="#94a3b8"
                  className="dark:fill-slate-400"
                />
              </marker>
              <marker
                id="arrowhead-hover"
                markerWidth="10"
                markerHeight="10"
                refX="9"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 10 3, 0 6" fill="#3b82f6" />
              </marker>
            </defs>

            {relations.map((relation, index) => {
              const sourcePos = nodePositions.get(relation.source);
              const targetPos = nodePositions.get(relation.target);
              if (!sourcePos || !targetPos) return null;

              const dx = targetPos.x - sourcePos.x;
              const dy = targetPos.y - sourcePos.y;
              const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
              const adjustedAngle = angle > 90 || angle < -90 ? angle + 180 : angle;

              const midX = (sourcePos.x + targetPos.x) / 2;
              const midY = (sourcePos.y + targetPos.y) / 2;

              return (
                <g key={`relation-${index}`}>
                  {/* Connection line */}
                  <line
                    x1={sourcePos.x}
                    y1={sourcePos.y}
                    x2={targetPos.x}
                    y2={targetPos.y}
                    stroke="#cbd5e1"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                    className="dark:stroke-slate-600 hover:stroke-blue-500 transition-colors"
                  />

                  {/* Rotated label */}
                  <text
                    x={midX}
                    y={midY - 8}
                    textAnchor="middle"
                    fontSize="12"
                    fill="#64748b"
                    className="dark:fill-slate-400 pointer-events-none select-none"
                    transform={`rotate(${adjustedAngle} ${midX} ${midY})`}
                  >
                    {relation.relation}
                  </text>
                </g>
              );
            })}

            {/* Render nodes */}
            {nodes.map((node) => {
              const pos = nodePositions.get(node);
              if (!pos) return null;
              const words = node.split(" ");
              const longest = Math.max(...words.map(w => w.length));

              const baseRadius = 30;
              const extra = Math.max(0, longest - 6) * 2.0;   // grow 2px for each char beyond 6
              const radius = baseRadius + extra;

              return (
                <g key={`node-${node}`}>
                  {/* Node circle */}
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={radius}
                    fill="#3b82f6"
                    className="hover:fill-blue-600 transition-colors cursor-grab active:cursor-grabbing"
                    onMouseDown={(e) => onMouseDownNode(node, e)}
                  />

                  {/* Node border */}
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={radius}
                    fill="none"
                    stroke="#1e40af"
                    strokeWidth="2"
                    className="opacity-0 hover:opacity-100 transition-opacity"
                  />

                  {/* Label */}
                  <text
                    x={pos.x}
                    y={pos.y}
                    textAnchor="middle"
                    dy="0.3em"
                    fontSize="10"
                    fontWeight="600"
                    fill="white"
                    pointerEvents="none"
                  >
                    {node.split(" ").map((word, i, arr) => (
                    <tspan
                      key={i}
                      x={pos.x}
                      dy={i === 0 ? `${-(arr.length - 1) * 0.6}em` : "1.2em"}
                    >
                      {word}
                    </tspan>
                  ))}
                  </text>
                </g>
              );
            })}
            </g>
          </svg>
        )}
      </div>

      {/* Legend */}
      {nodes.length > 0 && (
        <div className="flex-shrink-0 border-t border-slate-200 dark:border-slate-800 px-6 py-3 bg-white dark:bg-slate-900 text-xs text-slate-600 dark:text-slate-400">
          <p>• Blue circles represent nodes • Arrows show relations between nodes</p>
        </div>
      )}
    </div>
  );
}
