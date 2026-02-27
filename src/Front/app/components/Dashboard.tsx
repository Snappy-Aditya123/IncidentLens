import { useState } from 'react';
import { Link } from 'react-router';
import { useIncidents, useBackendHealth, useSeverityBreakdown, useMLAnomalies, useMLInfluencers } from '../hooks/useApi';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Skeleton } from './ui/skeleton';
import { 
  AlertTriangle, 
  Search, 
  Activity, 
  Shield, 
  TrendingUp,
  Clock,
  Database,
  Network,
  RefreshCw,
  Zap,
  Heart,
  BarChart3,
  Brain
} from 'lucide-react';
import { format } from 'date-fns';

export function Dashboard() {
  const [searchQuery, setSearchQuery] = useState('');
  const { data: incidents, loading, error, refetch } = useIncidents();
  const { data: health } = useBackendHealth();
  const { data: severityBreakdown, error: sevError } = useSeverityBreakdown();
  const { data: mlAnomalies, error: mlAnomalyError } = useMLAnomalies();
  const { data: mlInfluencers, error: mlInflError } = useMLInfluencers();

  const allIncidents = incidents ?? [];

  const filteredIncidents = allIncidents.filter(incident =>
    incident.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    incident.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const criticalCount = allIncidents.filter(i => i.severity === 'critical').length;
  const activeCount = allIncidents.filter(i => i.status === 'investigating').length;
  const avgAnomalyScore = allIncidents.length
    ? (allIncidents.reduce((sum, i) => sum + i.anomalyScore, 0) / allIncidents.length).toFixed(2)
    : '0.00';

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high': return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
      case 'medium': return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low': return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      default: return 'bg-slate-500/10 text-slate-500 border-slate-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'investigating': return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      case 'resolved': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'escalated': return 'bg-red-500/10 text-red-400 border-red-500/20';
      default: return 'bg-slate-500/10 text-slate-400 border-slate-500/20';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-slate-100">IncidentLens</h1>
                  {health ? (
                    <span className="flex items-center gap-1 text-xs" title={`Server: ${health.server}, ES: ${health.elasticsearch}`}>
                      <Heart className={`w-3 h-3 ${health.elasticsearch === 'green' || health.elasticsearch === 'yellow' ? 'text-green-400 fill-green-400' : 'text-red-400 fill-red-400'}`} />
                      <span className={health.elasticsearch === 'green' || health.elasticsearch === 'yellow' ? 'text-green-400' : 'text-red-400'}>
                        {health.error ? 'Degraded' : 'Online'}
                      </span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-xs" title="Backend unreachable">
                      <Heart className="w-3 h-3 text-slate-500" />
                      <span className="text-slate-500">Offline</span>
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-400">AI-Powered Network Investigation</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Link to="/simulation">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-emerald-700 text-emerald-400 hover:bg-emerald-900/30"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Simulation
                </Button>
              </Link>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-400">
                  {format(new Date(), 'PPpp')}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="text-slate-400 hover:text-slate-200"
                onClick={refetch}
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Active Incidents</CardTitle>
              <Activity className="w-4 h-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl text-slate-100">{activeCount}</div>
              <p className="text-xs text-slate-500 mt-1">Currently investigating</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Critical Alerts</CardTitle>
              <AlertTriangle className="w-4 h-4 text-red-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl text-slate-100">{criticalCount}</div>
              <p className="text-xs text-slate-500 mt-1">Require immediate attention</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Avg Anomaly Score</CardTitle>
              <TrendingUp className="w-4 h-4 text-orange-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl text-slate-100">{avgAnomalyScore}</div>
              <p className="text-xs text-slate-500 mt-1">Detection confidence</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Total Incidents</CardTitle>
              <Database className="w-4 h-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl text-slate-100">{allIncidents.length}</div>
              <p className="text-xs text-slate-500 mt-1">Last 24 hours</p>
            </CardContent>
          </Card>
        </div>

        {/* Severity Breakdown + ML Insights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {/* Severity Distribution */}
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Severity Distribution</CardTitle>
              <BarChart3 className="w-4 h-4 text-yellow-400" />
            </CardHeader>
            <CardContent>
              {severityBreakdown ? (
                <div className="space-y-2">
                  {Object.entries(severityBreakdown.severity_levels).map(([level, count]) => {
                    const pct = severityBreakdown.total_flows > 0
                      ? Math.round((count / severityBreakdown.total_flows) * 100) : 0;
                    const barColor = level === 'critical' ? 'bg-red-500' : level === 'high' ? 'bg-orange-500' : level === 'medium' ? 'bg-yellow-500' : 'bg-blue-500';
                    return (
                      <div key={level} className="flex items-center gap-2">
                        <span className="text-xs text-slate-400 w-16 capitalize">{level}</span>
                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div className={`h-full ${barColor} rounded-full`} style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-xs text-slate-300 w-12 text-right">{count}</span>
                      </div>
                    );
                  })}
                  <p className="text-xs text-slate-500 mt-1">{severityBreakdown.total_flows} total flows</p>
                </div>
              ) : sevError ? (
                <p className="text-xs text-red-400">Failed to load severity data</p>
              ) : (
                <p className="text-xs text-slate-500">{!health ? 'Start backend to see live data' : 'No data available'}</p>
              )}
            </CardContent>
          </Card>

          {/* ML Anomalies */}
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">ML Anomalies</CardTitle>
              <Brain className="w-4 h-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              {mlAnomalies && mlAnomalies.count > 0 ? (
                <div className="space-y-2">
                  <div className="text-2xl text-slate-100">{mlAnomalies.count}</div>
                  <p className="text-xs text-slate-500">High-score anomalies detected</p>
                  {mlAnomalies.records.slice(0, 3).map((rec, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 truncate max-w-[140px]">{rec.function}{rec.field_name ? ` (${rec.field_name})` : ''}</span>
                      <Badge variant="outline" className={rec.record_score >= 90 ? 'text-red-400 border-red-500/30' : rec.record_score >= 75 ? 'text-orange-400 border-orange-500/30' : 'text-yellow-400 border-yellow-500/30'}>
                        {Math.round(rec.record_score)}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div>
                  <div className="text-2xl text-slate-100">{mlAnomalyError ? '\u2014' : '0'}</div>
                  <p className={`text-xs ${mlAnomalyError ? 'text-red-400' : 'text-slate-500'}`}>
                    {mlAnomalyError
                      ? 'Failed to load ML anomalies'
                      : !health
                        ? 'Start backend to see ML data'
                        : 'No anomalies above threshold'}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* ML Influencers */}
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-slate-400">Top Influencers</CardTitle>
              <TrendingUp className="w-4 h-4 text-cyan-400" />
            </CardHeader>
            <CardContent>
              {mlInfluencers && mlInfluencers.count > 0 ? (
                <div className="space-y-2">
                  {mlInfluencers.influencers.slice(0, 5).map((inf, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-slate-400 truncate max-w-[120px]" title={`${inf.influencer_field_name}: ${inf.influencer_field_value}`}>
                        {inf.influencer_field_value}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${Math.min(inf.influencer_score, 100)}%` }} />
                        </div>
                        <span className="text-slate-300 w-8 text-right">{Math.round(inf.influencer_score)}</span>
                      </div>
                    </div>
                  ))}
                  <p className="text-xs text-slate-500 mt-1">{mlInfluencers.count} influencers found</p>
                </div>
              ) : mlInflError ? (
                <p className="text-xs text-red-400">Failed to load influencer data</p>
              ) : (
                <p className="text-xs text-slate-500">{!health ? 'Start backend to see influencer data' : 'No influencer data available'}</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Search and Filters */}
        <Card className="mb-6 bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-100">Incident Search</CardTitle>
            <CardDescription className="text-slate-400">
              Search and filter network security incidents
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <Search className="absolute left-3 top-3 w-4 h-4 text-slate-500" />
              <Input
                placeholder="Search by incident ID or title..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-slate-800 border-slate-700 text-slate-100 placeholder:text-slate-500"
              />
            </div>
          </CardContent>
        </Card>

        {/* Error banner */}
        {error && (
          <Card className="mb-6 bg-red-500/10 border-red-500/30">
            <CardContent className="py-4">
              <div className="flex items-center gap-3 text-red-400">
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">Backend unavailable — showing cached/mock data. {error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="space-y-4 mb-6">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="bg-slate-900 border-slate-800">
                <CardContent className="py-6">
                  <Skeleton className="h-5 w-2/3 mb-3 bg-slate-800" />
                  <Skeleton className="h-4 w-1/2 mb-2 bg-slate-800" />
                  <Skeleton className="h-3 w-1/3 bg-slate-800" />
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Incidents List */}
        {!loading && (
        <div className="space-y-4">
          {filteredIncidents.map((incident) => (
            <Card key={incident.id} className="bg-slate-900 border-slate-800 hover:border-slate-700 transition-colors">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <CardTitle className="text-slate-100">{incident.title}</CardTitle>
                      <Badge variant="outline" className={getSeverityColor(incident.severity)}>
                        {incident.severity.toUpperCase()}
                      </Badge>
                      <Badge variant="outline" className={getStatusColor(incident.status)}>
                        {incident.status}
                      </Badge>
                    </div>
                    <CardDescription className="text-slate-400">
                      {incident.description}
                    </CardDescription>
                  </div>
                  <Link to={`/investigation/${incident.id}`}>
                    <Button variant="outline" className="border-slate-700 text-slate-300 hover:bg-slate-800">
                      Investigate
                      <Network className="ml-2 w-4 h-4" />
                    </Button>
                  </Link>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-6 text-sm">
                  <div className="flex items-center gap-2 text-slate-400">
                    <Clock className="w-4 h-4" />
                    <span>{format(new Date(incident.timestamp), 'PPp')}</span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-400">
                    <Database className="w-4 h-4" />
                    <span>{incident.affectedSystems.length} affected systems</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-orange-400" />
                    <span className="text-slate-300">
                      Anomaly Score: <span className="text-orange-400">{(incident.anomalyScore * 100).toFixed(0)}%</span>
                    </span>
                  </div>
                </div>
                <div className="flex gap-2 mt-3 flex-wrap">
                  {incident.affectedSystems.map((system) => (
                    <Badge key={system} variant="secondary" className="bg-slate-800 text-slate-300 border-slate-700">
                      {system}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        )}

        {!loading && filteredIncidents.length === 0 && (
          <Card className="bg-slate-900 border-slate-800">
            <CardContent className="py-12">
              <div className="text-center text-slate-400">
                <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No incidents found matching your search.</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}