import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Upload, 
  HelpCircle, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle2, 
  X, 
  MessageSquare,
  ChevronRight,
  Info
} from 'lucide-react';
import { cn } from './lib/utils';

// --- Types ---

interface BinData {
  id: string;
  name: string;
  coverage: number;
  hits: number;
  predictedGain: number;
  recommendedSeed: string;
  score: number;
  explanation: string;
}

// --- Mock Data ---

const generateMockHeatmapData = (count: number): BinData[] => {
  return Array.from({ length: count }, (_, i) => ({
    id: `bin-${i}`,
    name: `Latent_Space_${i.toString(16).toUpperCase()}`,
    coverage: Math.random() < 0.2 ? 0 : Math.random() * 100,
    hits: Math.floor(Math.random() * 1000),
    predictedGain: Math.random() * 25,
    recommendedSeed: `0x${Math.floor(Math.random() * 16777215).toString(16).toUpperCase()}`,
    score: Math.floor(Math.random() * 100),
    explanation: "High density of unverified neural pathways detected in this cluster. New seed injection will normalize distribution."
  }));
};

const TOP_RECOMMENDATIONS: BinData[] = [
  {
    id: 'rec-1',
    name: 'Edge_Case_Alpha',
    coverage: 12.4,
    hits: 45,
    predictedGain: 24.2,
    recommendedSeed: '0xFD21',
    score: 98,
    explanation: "Critical boundary condition detected. This bin represents extreme input vectors that are currently under-sampled."
  },
  {
    id: 'rec-2',
    name: 'Vector_Quant_Niner',
    coverage: 45.2,
    hits: 210,
    predictedGain: 19.5,
    recommendedSeed: '0xAC19',
    score: 84,
    explanation: "Quantization noise is high in this region. Increasing coverage will improve model robustness against precision loss."
  },
  {
    id: 'rec-3',
    name: 'Temporal_Shift_X',
    coverage: 31.8,
    hits: 156,
    predictedGain: 15.1,
    recommendedSeed: '0xBB42',
    score: 72,
    explanation: "Time-series variance is peaking here. Recommended for stabilizing long-term sequence predictions."
  },
  {
    id: 'rec-4',
    name: 'Logic_Gate_7',
    coverage: 8.5,
    hits: 12,
    predictedGain: 11.8,
    recommendedSeed: '0x99FF',
    score: 61,
    explanation: "Deep logical branch with low visibility. Coverage here is essential for safety-critical decision paths."
  },
  {
    id: 'rec-5',
    name: 'Memory_Bank_Theta',
    coverage: 55.0,
    hits: 890,
    predictedGain: 8.4,
    recommendedSeed: '0xCC01',
    score: 49,
    explanation: "Recurrent state bottleneck identified. Optimizing this bin improves long-term context retention."
  }
];

// --- Components ---

const Tooltip = ({ data, visible }: { data: BinData | null, visible: boolean }) => {
  if (!data || !visible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: 10 }}
      className="absolute z-50 w-64 p-4 rounded-xl bg-slate-800/90 backdrop-blur-md border border-slate-700 shadow-2xl pointer-events-none"
      style={{ bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: '12px' }}
    >
      <div className="text-[10px] uppercase tracking-widest text-blue-400 font-bold mb-2">Bin: {data.name}</div>
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Coverage</span>
          <span className={cn("font-bold", data.coverage > 0 ? "text-green-400" : "text-slate-500")}>
            {data.coverage.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Hits</span>
          <span className="text-slate-200">{data.hits}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Predicted Gain</span>
          <span className="text-green-400">+{data.predictedGain.toFixed(1)}%</span>
        </div>
        <div className="pt-2 border-t border-slate-700/50">
          <div className="text-[10px] text-slate-400">Recommended Seed</div>
          <div className="text-xs font-mono text-blue-300">{data.recommendedSeed}</div>
        </div>
      </div>
    </motion.div>
  );
};

const HeatmapBin = ({ data }: { data: BinData }) => {
  const [isHovered, setIsHovered] = useState(false);

  const getBgColor = () => {
    if (data.coverage === 0) return 'bg-slate-800';
    if (data.coverage < 30) return 'bg-green-900/60';
    if (data.coverage < 70) return 'bg-green-600/80';
    return 'bg-green-400';
  };

  return (
    <div 
      className="relative"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <motion.div
        whileHover={{ scale: 1.4, zIndex: 10 }}
        className={cn(
          "w-3 h-3 rounded-sm cursor-crosshair transition-all duration-200",
          getBgColor(),
          isHovered && "shadow-[0_0_12px_rgba(74,225,118,0.6)]"
        )}
      />
      <AnimatePresence>
        {isHovered && <Tooltip data={data} visible={isHovered} />}
      </AnimatePresence>
    </div>
  );
};

const SummaryCard = ({ title, value, unit, icon: Icon, colorClass, subtitle }: any) => (
  <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 hover:bg-slate-800 transition-colors">
    <div className="flex items-center gap-3 mb-4">
      <div className={cn("p-2 rounded-lg bg-opacity-20", colorClass.replace('text-', 'bg-'))}>
        <Icon size={18} className={colorClass} />
      </div>
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
    </div>
    <div className="flex items-baseline gap-2">
      <span className="text-3xl font-bold text-slate-100">{value}</span>
      {unit && <span className={cn("text-lg font-bold", colorClass)}>{unit}</span>}
    </div>
    {subtitle && <p className="text-[10px] text-slate-500 mt-2 font-mono uppercase tracking-tighter">{subtitle}</p>}
  </div>
);

const RecommendationItem = ({ bin, index }: { bin: BinData, index: number }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className="group relative flex flex-col gap-3 p-4 rounded-xl hover:bg-slate-800/50 transition-all"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex justify-between items-center text-sm">
        <div className="flex items-center gap-3">
          <span className="font-mono text-slate-500 text-xs">#0{index + 1}</span>
          <span className="font-bold text-slate-200">{bin.name}</span>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden sm:block">
            <span className="text-[10px] text-slate-500 uppercase mr-2">Seed</span>
            <span className="text-xs font-mono text-blue-400">{bin.recommendedSeed}</span>
          </div>
          <div className="text-right">
            <span className="text-xs font-bold text-green-400">+{bin.predictedGain.toFixed(1)}% Gain</span>
          </div>
          <div className="bg-slate-700/50 px-2 py-0.5 rounded text-[10px] font-bold text-slate-300">
            SCORE: {bin.score}
          </div>
        </div>
      </div>
      
      <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${bin.score}%` }}
          transition={{ duration: 1, delay: index * 0.1 }}
          className="h-full bg-gradient-to-r from-green-500 to-green-400 shadow-[0_0_8px_rgba(74,225,118,0.3)]"
        />
      </div>

      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <p className="text-xs text-slate-400 leading-relaxed mt-1">
              {bin.explanation}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const FileUpload = () => {
  const [fileName, setFileName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (file.name.endsWith('.csv')) {
      setFileName(file.name);
    } else {
      alert('Please upload a CSV file.');
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div 
      className={cn(
        "relative h-full flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer group",
        isDragging ? "border-green-500 bg-green-500/5" : "border-slate-700 hover:border-slate-600 bg-slate-800/30",
        fileName && "border-green-500/50 bg-green-500/5"
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        accept=".csv"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      
      <div className={cn(
        "w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-transform group-hover:scale-110",
        fileName ? "bg-green-500/20" : "bg-slate-700/50"
      )}>
        {fileName ? (
          <CheckCircle2 className="text-green-400" size={32} />
        ) : (
          <Upload className="text-slate-400" size={32} />
        )}
      </div>

      <h3 className="text-lg font-bold text-slate-200 mb-2">
        {fileName ? "File Ready" : "Ingest Coverage Data"}
      </h3>
      <p className="text-sm text-slate-500 text-center max-w-[240px]">
        {fileName ? fileName : "Drag and drop your model's CSV trace files here for instant analysis."}
      </p>

      {!fileName && (
        <button className="mt-8 px-6 py-2.5 bg-green-500 hover:bg-green-400 text-slate-900 font-bold rounded-full text-sm transition-all shadow-lg shadow-green-500/20">
          Select Files
        </button>
      )}

      {fileName && (
        <button 
          onClick={(e) => { e.stopPropagation(); setFileName(null); }}
          className="mt-4 text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
        >
          <X size={12} /> Remove file
        </button>
      )}
    </div>
  );
};

const HelpPanel = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-8 right-8 z-[100] flex flex-col items-end gap-4">
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20, transformOrigin: 'bottom right' }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="w-80 bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 p-6 backdrop-blur-xl"
          >
            <div className="flex items-center justify-between mb-6">
              <h4 className="font-bold text-green-400 flex items-center gap-2">
                <MessageSquare size={18} />
                AI Assistant
              </h4>
              <button onClick={() => setIsOpen(false)} className="text-slate-500 hover:text-slate-300">
                <X size={18} />
              </button>
            </div>
            
            <div className="space-y-4 mb-6">
              <div className="bg-slate-700/30 p-3 rounded-xl">
                <p className="text-xs text-slate-300 leading-relaxed">
                  Hello! I can help you understand your model's coverage metrics. What would you like to know?
                </p>
              </div>
              <div className="space-y-2">
                <button className="w-full text-left p-2 rounded-lg hover:bg-slate-700/50 text-[11px] text-slate-400 flex items-center justify-between group">
                  How is coverage calculated?
                  <ChevronRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
                <button className="w-full text-left p-2 rounded-lg hover:bg-slate-700/50 text-[11px] text-slate-400 flex items-center justify-between group">
                  What are "latent space bins"?
                  <ChevronRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
                <button className="w-full text-left p-2 rounded-lg hover:bg-slate-700/50 text-[11px] text-slate-400 flex items-center justify-between group">
                  Explain the seed recommendations.
                  <ChevronRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
              </div>
            </div>

            <div className="relative">
              <input 
                type="text" 
                placeholder="Ask a question..." 
                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2 text-xs focus:outline-none focus:border-green-500/50 transition-colors"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-14 h-14 rounded-full flex items-center justify-center shadow-2xl transition-all hover:scale-105 active:scale-95",
          isOpen ? "bg-slate-700 text-slate-200" : "bg-blue-600 text-white"
        )}
      >
        {isOpen ? <X size={24} /> : <HelpCircle size={24} />}
      </button>
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [heatmapData] = useState(() => generateMockHeatmapData(480));

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 selection:bg-green-500/30">
      <div className="max-w-7xl mx-auto px-6 py-12 lg:py-16">
        
        {/* Header (Minimal) */}
        <header className="mb-12 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold text-green-500 tracking-tight">Neural Coverage Map</h1>
            <p className="text-slate-500 text-sm mt-1">Real-time latent space distribution analysis</p>
          </div>
          <div className="flex items-center gap-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            <span>Less</span>
            <div className="flex gap-1">
              <div className="w-3 h-3 rounded-sm bg-slate-800" />
              <div className="w-3 h-3 rounded-sm bg-green-900/60" />
              <div className="w-3 h-3 rounded-sm bg-green-600/80" />
              <div className="w-3 h-3 rounded-sm bg-green-400 shadow-[0_0_8px_rgba(74,225,118,0.4)]" />
            </div>
            <span>More</span>
          </div>
        </header>

        {/* 1. HEATMAP SECTION */}
        <section className="mb-12">
          <div className="bg-slate-800/30 p-8 rounded-3xl border border-slate-700/30">
            <div className="heatmap-grid">
              {heatmapData.map((bin) => (
                <HeatmapBin key={bin.id} data={bin} />
              ))}
            </div>
          </div>
        </section>

        {/* 2. SUMMARY BAR */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <SummaryCard 
            title="Total Coverage" 
            value="64.2" 
            unit="%" 
            icon={CheckCircle2} 
            colorClass="text-green-400"
            subtitle="Overall model visibility"
          />
          <SummaryCard 
            title="Uncovered Bins" 
            value="128" 
            icon={AlertCircle} 
            colorClass="text-red-400"
            subtitle="Critical gaps detected"
          />
          <SummaryCard 
            title="Predicted Improvement" 
            value="+18.4" 
            unit="%" 
            icon={TrendingUp} 
            colorClass="text-blue-400"
            subtitle="Next epoch estimate"
          />
        </section>

        {/* Bottom Grid: Recommendations & Upload */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
          
          {/* 3. RECOMMENDATION PANEL */}
          <section className="lg:col-span-8 bg-slate-800/30 rounded-3xl p-8 border border-slate-700/30">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-xl font-bold text-slate-100">Top 5 Optimization Targets</h2>
              <div className="flex items-center gap-2 text-[10px] font-bold text-blue-400 bg-blue-400/10 px-3 py-1 rounded-full border border-blue-400/20">
                <Info size={12} />
                ACTION REQUIRED
              </div>
            </div>
            <div className="space-y-2">
              {TOP_RECOMMENDATIONS.map((bin, idx) => (
                <RecommendationItem key={bin.id} bin={bin} index={idx} />
              ))}
            </div>
          </section>

          {/* 4. FILE UPLOAD */}
          <section className="lg:col-span-4">
            <FileUpload />
          </section>
        </div>

        {/* 5. FLOATING HELP */}
        <HelpPanel />

      </div>
    </div>
  );
}
