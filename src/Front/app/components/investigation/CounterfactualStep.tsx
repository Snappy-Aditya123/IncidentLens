import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Separator } from '../ui/separator';
import { Progress } from '../ui/progress';
import { Brain, ArrowRight, CheckCircle2, AlertCircle } from 'lucide-react';
import { CounterfactualExplanation } from '../../data/mockData';

interface CounterfactualStepProps {
  data: CounterfactualExplanation;
}

export function CounterfactualStep({ data }: CounterfactualStepProps) {
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

            {/* Insights */}
            <Card className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
              <CardHeader>
                <CardTitle className="text-slate-100">AI-Generated Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-slate-300">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>
                      <strong>Authentication method</strong> is the strongest indicator (92% impact). 
                      The direct database connection bypassing the Auth Service is highly suspicious.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>
                      <strong>Request rate</strong> of 450 req/min is 10x normal baseline. 
                      Reducing to normal levels would significantly lower threat classification.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>
                      <strong>IP rotation pattern</strong> suggests automated attack. 
                      Requests from a single known IP would reduce suspicion by 78%.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>
                      <strong>Off-hours access</strong> at 3:00 AM contributes moderately (45% impact). 
                      Business-hour access would be less suspicious.
                    </span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            {/* Recommendations */}
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">Recommended Actions</CardTitle>
                <CardDescription className="text-slate-400">
                  Based on counterfactual analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="space-y-3 text-sm text-slate-300">
                  <li className="flex gap-3">
                    <span className="text-blue-400 font-semibold flex-shrink-0">1.</span>
                    <span>Block direct database connections that bypass authentication service</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-blue-400 font-semibold flex-shrink-0">2.</span>
                    <span>Implement rate limiting: cap database queries to 50 requests/min per source</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-blue-400 font-semibold flex-shrink-0">3.</span>
                    <span>Add IP allowlisting for database access from internal services</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-blue-400 font-semibold flex-shrink-0">4.</span>
                    <span>Enhance monitoring for off-hours database access patterns</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="text-blue-400 font-semibold flex-shrink-0">5.</span>
                    <span>Require MFA for all database administrative operations</span>
                  </li>
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
            <p className="text-sm text-slate-400">
              All analysis steps completed. Review the findings and implement recommended actions.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
