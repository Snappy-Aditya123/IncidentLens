import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ChevronRight, Network as NetworkIcon, TrendingUp } from 'lucide-react';
import { NetworkNode, NetworkEdge } from '../../data/mockData';

interface GNNStepProps {
  data: {
    nodes: NetworkNode[];
    edges: NetworkEdge[];
  };
  onNext: () => void;
}

export function GNNStep({ data, onNext }: GNNStepProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 800;
    const height = 500;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Create arrow markers
    svg.append('defs').selectAll('marker')
      .data(['normal', 'anomalous'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'anomalous' ? '#ef4444' : '#64748b');

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes as any)
      .force('link', d3.forceLink(data.edges)
        .id((d: any) => d.id)
        .distance(150))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));

    // Draw edges
    const link = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', d => d.anomalous ? '#ef4444' : '#475569')
      .attr('stroke-width', d => d.weight / 2)
      .attr('stroke-opacity', d => d.anomalous ? 0.8 : 0.3)
      .attr('marker-end', d => `url(#arrow-${d.anomalous ? 'anomalous' : 'normal'})`);

    // Draw nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(data.nodes)
      .join('g')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    // Node circles
    node.append('circle')
      .attr('r', 20)
      .attr('fill', d => {
        switch (d.status) {
          case 'compromised': return '#dc2626';
          case 'suspicious': return '#f59e0b';
          case 'normal': return '#10b981';
          default: return '#64748b';
        }
      })
      .attr('stroke', d => {
        switch (d.status) {
          case 'compromised': return '#ef4444';
          case 'suspicious': return '#fbbf24';
          case 'normal': return '#34d399';
          default: return '#94a3b8';
        }
      })
      .attr('stroke-width', 3);

    // Node labels
    node.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', 35)
      .attr('fill', '#cbd5e1')
      .attr('font-size', '11px');

    // Risk indicators
    node.append('text')
      .text(d => `${(d.risk * 100).toFixed(0)}%`)
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [data]);

  const compromisedNodes = data.nodes.filter(n => n.status === 'compromised').length;
  const suspiciousNodes = data.nodes.filter(n => n.status === 'suspicious').length;
  const anomalousEdges = data.edges.filter(e => e.anomalous).length;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-slate-400">Compromised Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-red-400">{compromisedNodes}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-slate-400">Suspicious Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-orange-400">{suspiciousNodes}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-slate-400">Anomalous Connections</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-yellow-400">{anomalousEdges}</div>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-slate-100 flex items-center gap-2">
                <NetworkIcon className="w-5 h-5" />
                Network Graph Analysis (GNN)
              </CardTitle>
              <CardDescription className="text-slate-400">
                Graph Neural Network analysis of system relationships
              </CardDescription>
            </div>
            <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
              {data.nodes.length} nodes • {data.edges.length} edges
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
            <svg ref={svgRef} className="w-full" />
          </div>

          <div className="mt-6 flex gap-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-600 border-2 border-red-400" />
              <span className="text-sm text-slate-400">Compromised</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-orange-600 border-2 border-orange-400" />
              <span className="text-sm text-slate-400">Suspicious</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-600 border-2 border-green-400" />
              <span className="text-sm text-slate-400">Normal</span>
            </div>
            <div className="flex items-center gap-2 ml-auto">
              <div className="w-8 h-0.5 bg-red-500" />
              <span className="text-sm text-slate-400">Anomalous Connection</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            GNN Risk Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {data.nodes
              .sort((a, b) => b.risk - a.risk)
              .slice(0, 5)
              .map((node) => (
                <div key={node.id} className="flex items-center justify-between bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      node.status === 'compromised' ? 'bg-red-500' :
                      node.status === 'suspicious' ? 'bg-orange-500' :
                      'bg-green-500'
                    }`} />
                    <span className="text-slate-300">{node.label}</span>
                    <Badge variant="secondary" className="bg-slate-800 text-slate-400 text-xs">
                      {node.type}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-sm text-slate-400">Risk Score:</div>
                    <div className={`text-sm font-medium ${
                      node.risk > 0.8 ? 'text-red-400' :
                      node.risk > 0.5 ? 'text-orange-400' :
                      'text-green-400'
                    }`}>
                      {(node.risk * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-gradient-to-br from-green-500/10 to-blue-500/10 border-green-500/20">
        <CardContent className="py-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-slate-100 mb-2">Network Analysis Complete</h3>
              <p className="text-sm text-slate-400">
                Generate counterfactual explanations to understand the incident
              </p>
            </div>
            <Button 
              onClick={onNext}
              className="bg-green-600 hover:bg-green-700"
            >
              Generate Explanations
              <ChevronRight className="ml-2 w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
