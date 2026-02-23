import { useState } from 'react';
import { useParams, Link } from 'react-router';
import { useIncident, useElasticsearchData, useNetworkGraph, useCounterfactual } from '../hooks/useApi';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { 
  ArrowLeft, 
  Database, 
  Network, 
  Brain,
  ChevronRight,
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { ElasticsearchStep } from './investigation/ElasticsearchStep';
import { GNNStep } from './investigation/GNNStep';
import { CounterfactualStep } from './investigation/CounterfactualStep';
import { format } from 'date-fns';

type InvestigationStep = 'overview' | 'elasticsearch' | 'gnn' | 'counterfactual';

export function Investigation() {
  const { incidentId } = useParams();
  const [currentStep, setCurrentStep] = useState<InvestigationStep>('overview');
  
  const { data: incident, loading: incidentLoading } = useIncident(incidentId);
  const { data: elasticsearchData, loading: esLoading } = useElasticsearchData(incidentId);
  const { data: networkGraph, loading: graphLoading } = useNetworkGraph(incidentId);
  const { data: counterfactual, loading: cfLoading } = useCounterfactual(incidentId);

  if (incidentLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
          <p className="text-slate-400">Loading investigation…</p>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="py-12">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
            <p className="text-center text-slate-300">Incident not found</p>
            <div className="flex justify-center mt-6">
              <Link to="/">
                <Button variant="outline" className="border-slate-700">
                  <ArrowLeft className="mr-2 w-4 h-4" />
                  Back to Dashboard
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const steps = [
    { id: 'overview', label: 'Overview', icon: Clock, completed: true },
    { id: 'elasticsearch', label: 'Log Analysis', icon: Database, completed: currentStep !== 'overview' },
    { id: 'gnn', label: 'Network Graph', icon: Network, completed: currentStep === 'counterfactual' },
    { id: 'counterfactual', label: 'Explainability', icon: Brain, completed: false },
  ];

  const getCurrentStepIndex = () => {
    return steps.findIndex(s => s.id === currentStep);
  };

  const progress = ((getCurrentStepIndex() + 1) / steps.length) * 100;

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high': return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
      case 'medium': return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low': return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      default: return 'bg-slate-500/10 text-slate-500 border-slate-500/20';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/">
                <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-300">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
              </Link>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-slate-100">{incident.title}</h1>
                  <Badge variant="outline" className={getSeverityColor(incident.severity)}>
                    {incident.severity.toUpperCase()}
                  </Badge>
                </div>
                <p className="text-sm text-slate-400 mt-1">
                  Incident ID: {incident.id} • {format(new Date(incident.timestamp), 'PPpp')}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Steps */}
      <div className="border-b border-slate-800 bg-slate-900/30">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => {
              const StepIcon = step.icon;
              const isActive = step.id === currentStep;
              const isCompleted = step.completed;
              
              return (
                <div key={step.id} className="flex items-center flex-1">
                  <button
                    onClick={() => setCurrentStep(step.id as InvestigationStep)}
                    className={`flex items-center gap-3 px-4 py-2 rounded-lg transition-all ${
                      isActive 
                        ? 'bg-blue-500/20 border border-blue-500/30' 
                        : isCompleted
                        ? 'bg-slate-800/50 hover:bg-slate-800 border border-slate-700'
                        : 'bg-slate-800/30 border border-slate-800'
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      isActive 
                        ? 'bg-blue-500 text-white' 
                        : isCompleted
                        ? 'bg-green-500 text-white'
                        : 'bg-slate-700 text-slate-400'
                    }`}>
                      {isCompleted && !isActive ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : (
                        <StepIcon className="w-4 h-4" />
                      )}
                    </div>
                    <div className="text-left">
                      <div className={`text-sm ${isActive ? 'text-slate-100' : 'text-slate-400'}`}>
                        {step.label}
                      </div>
                    </div>
                  </button>
                  {index < steps.length - 1 && (
                    <ChevronRight className="w-5 h-5 text-slate-600 mx-2" />
                  )}
                </div>
              );
            })}
          </div>
          <Progress value={progress} className="h-2 bg-slate-800" />
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        {currentStep === 'overview' && (
          <div className="space-y-6">
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Incident Overview</CardTitle>
                <CardDescription className="text-slate-400">
                  Initial details and affected systems
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm text-slate-400 mb-2">Description</h3>
                    <p className="text-slate-200">{incident.description}</p>
                  </div>
                  
                  <Separator className="bg-slate-800" />
                  
                  <div>
                    <h3 className="text-sm text-slate-400 mb-2">Affected Systems</h3>
                    <div className="flex gap-2 flex-wrap">
                      {incident.affectedSystems.map((system) => (
                        <Badge key={system} variant="secondary" className="bg-slate-800 text-slate-300 border-slate-700">
                          {system}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <Separator className="bg-slate-800" />
                  
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <h3 className="text-sm text-slate-400 mb-2">Anomaly Score</h3>
                      <div className="text-2xl text-orange-400">
                        {(incident.anomalyScore * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm text-slate-400 mb-2">Status</h3>
                      <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
                        {incident.status}
                      </Badge>
                    </div>
                    <div>
                      <h3 className="text-sm text-slate-400 mb-2">Timestamp</h3>
                      <div className="text-slate-300">
                        {format(new Date(incident.timestamp), 'HH:mm:ss')}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
              <CardContent className="py-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-slate-100 mb-2">Ready to begin investigation?</h3>
                    <p className="text-sm text-slate-400">
                      Start with Elasticsearch log analysis to gather evidence
                    </p>
                  </div>
                  <Button 
                    onClick={() => setCurrentStep('elasticsearch')}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Start Investigation
                    <ChevronRight className="ml-2 w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {currentStep === 'elasticsearch' && (
          esLoading ? (
            <StepLoading label="Loading Elasticsearch data…" />
          ) : elasticsearchData ? (
            <ElasticsearchStep 
              data={elasticsearchData} 
              onNext={() => setCurrentStep('gnn')}
            />
          ) : <StepEmpty label="No Elasticsearch data available for this incident." />
        )}

        {currentStep === 'gnn' && (
          graphLoading ? (
            <StepLoading label="Building network graph…" />
          ) : networkGraph ? (
            <GNNStep 
              data={networkGraph}
              onNext={() => setCurrentStep('counterfactual')}
            />
          ) : <StepEmpty label="No network graph data available for this incident." />
        )}

        {currentStep === 'counterfactual' && (
          cfLoading ? (
            <StepLoading label="Computing counterfactual explanations…" />
          ) : counterfactual ? (
            <CounterfactualStep data={counterfactual} />
          ) : <StepEmpty label="No counterfactual data available for this incident." />
        )}
      </div>
    </div>
  );
}

/* ─── tiny loading/empty helpers ───────────── */

function StepLoading({ label }: { label: string }) {
  return (
    <Card className="bg-slate-900 border-slate-800">
      <CardContent className="py-16 text-center">
        <Loader2 className="w-10 h-10 mx-auto mb-4 text-blue-400 animate-spin" />
        <p className="text-slate-400">{label}</p>
      </CardContent>
    </Card>
  );
}

function StepEmpty({ label }: { label: string }) {
  return (
    <Card className="bg-slate-900 border-slate-800">
      <CardContent className="py-16 text-center">
        <AlertCircle className="w-10 h-10 mx-auto mb-4 text-slate-500" />
        <p className="text-slate-400">{label}</p>
      </CardContent>
    </Card>
  );
}
