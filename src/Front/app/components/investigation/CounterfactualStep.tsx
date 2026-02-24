import { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Separator } from '../ui/separator';
import { Progress } from '../ui/progress';
import { Brain, ArrowRight, CheckCircle2, AlertCircle, Home } from 'lucide-react';
import { CounterfactualExplanation } from '../../data/mockData';
import { Link } from 'react-router';

interface CounterfactualStepProps {
  data: CounterfactualExplanation;
}

/** Derive human-readable insights from actual counterfactual changes. */
function deriveInsights(changes: CounterfactualExplanation['changes']): string[] {
  if (!changes || changes.length === 0) {
    return ['No significant feature differences were detected between the anomalous and normal flows.'];
  }

  const sorted = [...changes].sort((a, b) => b.impact - a.impact);
  return sorted.slice(0, 4).map((c) => {
    const pct = (c.impact * 100).toFixed(0);
    return `**${c.parameter}** changed from ${c.original} → ${c.modified} (${pct}% impact). ` +
      `${c.impact > 0.7 ? 'This is a primary driver of the anomaly classification.' : 'This contributes to the deviation from normal traffic.'}`;
  });
}

/** Derive actionable recommendations from actual counterfactual changes. */
function deriveRecommendations(changes: CounterfactualExplanation['changes']): string[] {
  if (!changes || changes.length === 0) {
    return ['Continue monitoring for anomalous traffic patterns.'];
  }

  const sorted = [...changes].sort((a, b) => b.impact - a.impact);
  const recs: string[] = [];

  for (const c of sorted.slice(0, 5)) {
    const param = c.parameter.toLowerCase();
    if (param.includes('packet') || param.includes('count')) {
      recs.push(`Investigate abnormal packet volume for ${c.parameter} (${c.original} → expected ${c.modified}). Consider rate limiting.`);
    } else if (param.includes('byte') || param.includes('payload')) {
      recs.push(`Examine payload sizes: ${c.parameter} shows ${c.original} vs normal ${c.modified}. May indicate data exfiltration or amplification.`);
    } else if (param.includes('iat') || param.includes('time') || param.includes('interval')) {
      recs.push(`Review timing patterns: ${c.parameter} (${c.original} vs ${c.modified}). Unusual intervals may indicate automated activity.`);
    } else {
      recs.push(`Monitor ${c.parameter}: current value ${c.original} deviates significantly from normal baseline ${c.modified}.`);
    }
  }

  return recs;
}

export function CounterfactualStep({ data }: CounterfactualStepProps) {
  const insights = useMemo(() => deriveInsights(data.changes), [data.changes]);
  const recommendations = useMemo(() => deriveRecommendations(data.changes), [data.changes]);

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100 flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Counterfactual Explainability Analysis
          </CardTitle>
          <CardDescription className="text-slate-400">
            Understanding what factors contribute to the incident classification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Scenario Comparison */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <h3 className="text-slate-100">Current Scenario</h3>
                </div>
                <p className="text-sm text-slate-300">{data.original}</p>
                <div className="mt-4">
                  <Badge variant="outline" className="bg-red-500/20 text-red-400 border-red-500/30">
                    {data.prediction.original}
                  </Badge>
                </div>
              </div>

              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                  <h3 className="text-slate-100">Counterfactual Scenario</h3>
                </div>
                <p className="text-sm text-slate-300">{data.counterfactual}</p>
                <div className="mt-4">
                  <Badge variant="outline" className="bg-green-500/20 text-green-400 border-green-500/30">
                    {data.prediction.counterfactual}
                  </Badge>
                </div>
              </div>
            </div>

            <Separator className="bg-slate-800" />

            {/* Key Changes */}
            <div>
              <h3 className="text-slate-100 mb-4 flex items-center gap-2">
                <ArrowRight className="w-5 h-5" />
                Key Factors for Classification Change
              </h3>
              <div className="space-y-4">
                {data.changes.map((change, index) => (
                  <div 
                    key={index}
                    className="bg-slate-950 border border-slate-800 rounded-lg p-4"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="text-slate-200 mb-1">{change.parameter}</h4>
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-red-400">{change.original}</span>
                          <ArrowRight className="w-4 h-4 text-slate-500" />
                          <span className="text-green-400">{change.modified}</span>
                        </div>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={`${
                          change.impact > 0.8 ? 'bg-red-500/20 text-red-400 border-red-500/30' :
                          change.impact > 0.6 ? 'bg-orange-500/20 text-orange-400 border-orange-500/30' :
                          'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                        }`}
                      >
                        Impact: {(change.impact * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Contribution to classification</span>
                        <span>{(change.impact * 100).toFixed(0)}%</span>
                      </div>
                      <Progress 
                        value={change.impact * 100} 
                        className="h-2 bg-slate-800"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <Separator className="bg-slate-800" />

            {/* Dynamic Insights derived from actual data */}
            <Card className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
              <CardHeader>
                <CardTitle className="text-slate-100">Analysis Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-slate-300">
                  {insights.map((insight, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                      <span dangerouslySetInnerHTML={{ __html: insight.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Dynamic Recommendations derived from actual data */}
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Recommended Actions</CardTitle>
                <CardDescription className="text-slate-400">
                  Based on counterfactual analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="space-y-3 text-sm text-slate-300">
                  {recommendations.map((rec, i) => (
                    <li key={i} className="flex gap-3">
                      <span className="text-blue-400 font-semibold flex-shrink-0">{i + 1}.</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ol>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
        <CardContent className="py-6">
          <div className="text-center">
            <CheckCircle2 className="w-12 h-12 mx-auto mb-4 text-green-400" />
            <h3 className="text-slate-100 mb-2">Investigation Complete</h3>
            <p className="text-sm text-slate-400 mb-4">
              All analysis steps completed. Review the findings and implement recommended actions.
            </p>
            <Link to="/">
              <Button className="bg-green-600 hover:bg-green-700">
                <Home className="mr-2 w-4 h-4" />
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
