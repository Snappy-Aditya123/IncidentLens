import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Separator } from '../ui/separator';
import { ChevronRight, Terminal, FileCode } from 'lucide-react';

interface ElasticsearchStepProps {
  data: {
    totalHits: number;
    logs: Array<{
      timestamp: string;
      source: string;
      message: string;
      level: string;
    }>;
    query: any;
  };
  onNext: () => void;
}

export function ElasticsearchStep({ data, onNext }: ElasticsearchStepProps) {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'ERROR': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'WARNING': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'INFO': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
  };

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-slate-100 flex items-center gap-2">
                <Terminal className="w-5 h-5" />
                Elasticsearch Log Analysis
              </CardTitle>
              <CardDescription className="text-slate-400">
                Analyzing system logs and event data
              </CardDescription>
            </div>
            <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20">
              {data.totalHits.toLocaleString()} hits
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-slate-950 rounded-lg p-4 border border-slate-800">
              <div className="flex items-center gap-2 mb-3">
                <FileCode className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm text-slate-300">Query DSL</h3>
              </div>
              <pre className="text-xs text-slate-400 overflow-x-auto">
                {JSON.stringify(data.query, null, 2)}
              </pre>
            </div>

            <Separator className="bg-slate-800" />

            <div>
              <h3 className="text-sm text-slate-300 mb-4">Recent Log Entries</h3>
              <div className="space-y-3">
                {data.logs.map((log, index) => (
                  <div 
                    key={index}
                    className="bg-slate-950 rounded-lg p-4 border border-slate-800 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={getLevelColor(log.level)}>
                          {log.level}
                        </Badge>
                        <span className="text-xs text-slate-500">{log.source}</span>
                      </div>
                      <span className="text-xs text-slate-500">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm text-slate-300">{log.message}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/20">
        <CardContent className="py-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-slate-100 mb-2">Log Analysis Complete</h3>
              <p className="text-sm text-slate-400">
                Proceed to network graph analysis using Graph Neural Networks
              </p>
            </div>
            <Button 
              onClick={onNext}
              className="bg-purple-600 hover:bg-purple-700"
            >
              Analyze Network Graph
              <ChevronRight className="ml-2 w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
