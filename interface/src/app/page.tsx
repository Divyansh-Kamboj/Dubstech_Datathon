"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BRIEF_TEXT =
  "The year is 2030. The system has reached its breaking point. You have been appointed Director of the Regional Safety Net. Your budget is $150 Million. Every dollar you spend on one life is a dollar stolen from another. History will only remember the body count.";

const EFFICIENCY_TEXT =
  "Efficiency determines the cost-to-life ratio. A high weight prioritizes treatments that save the most people for the least amount of money.";
const HUMANITY_TEXT =
  "Humanity weight shifts focus to clinical urgency. A high weight prioritizes the sickest patients with the highest immediate mortality risk, regardless of cost.";
const YES_OUTCOME_TEXT =
  "Project Aesclepius was originally developed for the DubsTech Datathon. This gamified interface is a narrative layer built over our real-world clinical allocation engine. To see the raw data, ML model metrics, and the full technical dashboard, visit our primary interface below.";

type SimResponse = {
  lives_covered: number;
  unmet_need: number;
  budget_remaining: number;
  chart_data: Array<{
    Department: string;
    Funded_Pct: number;
  }>;
};

export default function Home() {
  const [step, setStep] = useState(1);
  const [wEfficiency, setWEfficiency] = useState(0.5);
  const [wHumanity, setWHumanity] = useState(0.5);
  const [typedCount, setTypedCount] = useState(0);
  const [typedWeightCount, setTypedWeightCount] = useState(0);
  const [typedOutcomeCount, setTypedOutcomeCount] = useState(0);
  const [isFetching, setIsFetching] = useState(false);
  const [simulation, setSimulation] = useState<SimResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resetRun = () => {
    setWEfficiency(0.5);
    setWHumanity(0.5);
    setSimulation(null);
    setError(null);
    setStep(1);
  };

  const typedBrief = useMemo(
    () => BRIEF_TEXT.slice(0, typedCount),
    [typedCount],
  );
  const weightNarrative = step === 3 ? EFFICIENCY_TEXT : HUMANITY_TEXT;
  const typedWeightNarrative = useMemo(
    () => weightNarrative.slice(0, typedWeightCount),
    [typedWeightCount, weightNarrative],
  );
  const typedOutcomeNarrative = useMemo(
    () => YES_OUTCOME_TEXT.slice(0, typedOutcomeCount),
    [typedOutcomeCount],
  );
  const chartData = useMemo(() => {
    if (!simulation) {
      return [];
    }

    const grouped = new Map<
      string,
      { coveredPatients: number; uncoveredPatients: number }
    >();

    for (const row of simulation.chart_data) {
      const current = grouped.get(row.Department) ?? {
        coveredPatients: 0,
        uncoveredPatients: 0,
      };
      current.coveredPatients += row.Funded_Pct * 100;
      current.uncoveredPatients += (1 - row.Funded_Pct) * 100;
      grouped.set(row.Department, current);
    }

    return Array.from(grouped.entries()).map(([Department, values]) => ({
      Department,
      coveredPatients: Math.round(values.coveredPatients),
      uncoveredPatients: Math.round(values.uncoveredPatients),
    }));
  }, [simulation]);

  useEffect(() => {
    if (step !== 2) {
      setTypedCount(0);
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let cursor = 0;

    const tick = () => {
      if (cancelled) {
        return;
      }

      cursor += 1;
      setTypedCount(cursor);

      if (cursor >= BRIEF_TEXT.length) {
        return;
      }

      const typedSoFar = BRIEF_TEXT.slice(0, cursor);
      const previousChar = BRIEF_TEXT[cursor - 1];

      let delay = 40;
      if (typedSoFar === "The year is 2030.") {
        delay = 750;
      } else if (previousChar === ".") {
        delay = 400;
      }

      timeoutId = setTimeout(tick, delay);
    };

    timeoutId = setTimeout(tick, 40);

    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [step]);

  useEffect(() => {
    if (step !== 3 && step !== 4) {
      setTypedWeightCount(0);
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let cursor = 0;

    const tick = () => {
      if (cancelled) {
        return;
      }

      cursor += 1;
      setTypedWeightCount(cursor);

      if (cursor >= weightNarrative.length) {
        return;
      }

      timeoutId = setTimeout(tick, 28);
    };

    timeoutId = setTimeout(tick, 28);
    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [step, weightNarrative]);

  useEffect(() => {
    if (step !== 7) {
      setTypedOutcomeCount(0);
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let cursor = 0;

    const tick = () => {
      if (cancelled) {
        return;
      }

      cursor += 1;
      setTypedOutcomeCount(cursor);

      if (cursor >= YES_OUTCOME_TEXT.length) {
        return;
      }

      const previousChar = YES_OUTCOME_TEXT[cursor - 1];
      const delay = previousChar === "." ? 450 : 35;
      timeoutId = setTimeout(tick, delay);
    };

    timeoutId = setTimeout(tick, 35);

    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [step]);

  useEffect(() => {
    if (step !== 5 || simulation || isFetching) {
      return;
    }

    const runSimulation = async () => {
      setIsFetching(true);
      setError(null);
      try {
        const response = await fetch("http://localhost:8000/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            w_efficiency: wEfficiency,
            w_humanity: wHumanity,
          }),
        });

        if (!response.ok) {
          throw new Error(`Simulation request failed (${response.status})`);
        }

        const payload = (await response.json()) as SimResponse;
        setSimulation(payload);
      } catch (err) {
        console.error("Simulation fetch error:", err);
        const message =
          err instanceof Error ? err.message : "Unknown simulation error";
        setError(message);
      } finally {
        setIsFetching(false);
      }
    };

    void runSimulation();
  }, [step, simulation, isFetching, wEfficiency, wHumanity]);

  return (
    <main className="flex min-h-screen w-full flex-col justify-center px-6 py-10 pb-24 md:px-10">
      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.section
            key="step-1"
            className="relative flex min-h-screen w-full items-center justify-center text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.45 }}
          >
            <h1 className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[165%] text-6xl leading-none text-black md:text-8xl">
              Project Aesclepius
            </h1>
            <p
              role="button"
              tabIndex={0}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 translate-y-[165%] cursor-pointer text-2xl uppercase underline underline-offset-8 transition-transform duration-200 hover:scale-105"
              onClick={() => setStep(2)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  setStep(2);
                }
              }}
            >
              Click to Continue
            </p>
          </motion.section>
        )}

        {step === 2 && (
          <motion.section
            key="step-2"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 px-4 text-center"
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -24 }}
            transition={{ duration: 0.4 }}
          >
            <p className="typewriter-narrative max-w-5xl text-center text-2xl leading-relaxed md:text-3xl">
              {typedBrief}
              <span className="typewriter-cursor" />
            </p>
            <p
              role="button"
              tabIndex={typedCount < BRIEF_TEXT.length ? -1 : 0}
              className={`text-2xl uppercase underline underline-offset-8 transition-transform duration-200 ${
                typedCount < BRIEF_TEXT.length
                  ? "pointer-events-none opacity-35"
                  : "cursor-pointer hover:scale-105"
              }`}
              onClick={() => setStep(3)}
              onKeyDown={(event) => {
                if ((event.key === "Enter" || event.key === " ") && typedCount >= BRIEF_TEXT.length) {
                  setStep(3);
                }
              }}
            >
              Continue
            </p>
          </motion.section>
        )}

        {step === 3 && (
          <motion.section
            key="step-3"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 p-8 text-center md:p-12"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -24 }}
            transition={{ duration: 0.4 }}
          >
            <p className="typewriter-narrative max-w-5xl text-center text-2xl leading-relaxed md:text-3xl">
              {typedWeightNarrative}
              <span className="typewriter-cursor" />
            </p>
            <div className="flex flex-col items-center gap-4">
              <p className="text-center text-3xl uppercase">Efficiency</p>
              <input
                className="horizontal-sleek-slider"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={wEfficiency}
                onChange={(event) =>
                  setWEfficiency(Number.parseFloat(event.target.value))
                }
              />
              <p className="text-xl">{Math.round(wEfficiency * 100)}%</p>
            </div>
            <p
              role="button"
              tabIndex={typedWeightCount < EFFICIENCY_TEXT.length ? -1 : 0}
              className={`text-2xl uppercase underline underline-offset-8 transition-transform duration-200 ${
                typedWeightCount < EFFICIENCY_TEXT.length
                  ? "pointer-events-none opacity-35"
                  : "cursor-pointer hover:scale-105"
              }`}
              onClick={() => setStep(4)}
              onKeyDown={(event) => {
                if ((event.key === "Enter" || event.key === " ") && typedWeightCount >= EFFICIENCY_TEXT.length) {
                  setStep(4);
                }
              }}
            >
              Next
            </p>
          </motion.section>
        )}

        {step === 4 && (
          <motion.section
            key="step-4"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 p-8 text-center md:p-12"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -24 }}
            transition={{ duration: 0.4 }}
          >
            <p className="typewriter-narrative max-w-5xl text-center text-2xl leading-relaxed md:text-3xl">
              {typedWeightNarrative}
              <span className="typewriter-cursor" />
            </p>
            <div className="flex flex-col items-center gap-4">
              <p className="text-center text-3xl uppercase">Humanity</p>
              <input
                className="horizontal-sleek-slider"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={wHumanity}
                onChange={(event) =>
                  setWHumanity(Number.parseFloat(event.target.value))
                }
              />
              <p className="text-xl">{Math.round(wHumanity * 100)}%</p>
            </div>
            <p
              role="button"
              tabIndex={typedWeightCount < HUMANITY_TEXT.length ? -1 : 0}
              className={`text-2xl uppercase underline underline-offset-8 transition-transform duration-200 ${
                typedWeightCount < HUMANITY_TEXT.length
                  ? "pointer-events-none opacity-35"
                  : "cursor-pointer hover:scale-105"
              }`}
              onClick={() => {
                setSimulation(null);
                setError(null);
                setStep(5);
              }}
              onKeyDown={(event) => {
                if ((event.key === "Enter" || event.key === " ") && typedWeightCount >= HUMANITY_TEXT.length) {
                  setSimulation(null);
                  setError(null);
                  setStep(5);
                }
              }}
            >
              Simulate
            </p>
          </motion.section>
        )}

        {step === 5 && (
          <motion.section
            key="step-5"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 p-8 text-center md:p-12"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -18 }}
            transition={{ duration: 0.4 }}
          >
            {error && <p className="text-xl uppercase">Simulation Error: {error}</p>}

            {!error && !simulation && isFetching && (
              <div className="h-[320px] w-full max-w-5xl animate-pulse" />
            )}

            {simulation && (
              <>
                <div className="grid w-full max-w-4xl gap-6 md:grid-cols-2">
                  <div>
                    <p className="text-sm uppercase tracking-[0.2em]">Lives Covered</p>
                    <p className="mt-2 text-5xl text-black md:text-7xl">
                      {simulation.lives_covered.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm uppercase tracking-[0.2em]">Death Toll</p>
                    <p className="mt-2 text-5xl text-black md:text-7xl">
                      {simulation.unmet_need.toLocaleString()}
                    </p>
                  </div>
                </div>

                <section className="w-full max-w-5xl">
                  <div className="h-[340px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 16, right: 16, left: 4, bottom: 16 }}>
                        <XAxis
                          dataKey="Department"
                          axisLine={false}
                          tickLine={false}
                          tick={{
                            fill: "#000000",
                            fontSize: 12,
                            fontFamily: "var(--font-special-elite)",
                          }}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tick={{
                            fill: "#000000",
                            fontSize: 12,
                            fontFamily: "var(--font-special-elite)",
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#ff0000",
                            border: "none",
                            color: "#000000",
                            fontFamily: "var(--font-special-elite)",
                          }}
                          labelStyle={{ color: "#000000" }}
                          // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          formatter={(value: any, name: any) => [
                            value ? value.toLocaleString() : "0",
                            name === "coveredPatients"
                              ? "Covered Patients"
                              : "Unmet Need",
                          ]}
                        />
                        <Bar
                          dataKey="coveredPatients"
                          stackId="coverage"
                          fill="#000000"
                          isAnimationActive
                          animationDuration={1500}
                          animationBegin={0}
                          animationEasing="ease-out"
                        />
                        <Bar
                          dataKey="uncoveredPatients"
                          stackId="coverage"
                          fill="#2a2a2a"
                          isAnimationActive
                          animationDuration={1500}
                          animationBegin={0}
                          animationEasing="ease-out"
                        />
                        <Legend
                          verticalAlign="top"
                          align="center"
                          iconType="square"
                          wrapperStyle={{
                            fontFamily: "var(--font-special-elite)",
                            color: "#000000",
                            fontSize: "12px",
                            paddingBottom: "10px",
                          }}
                          formatter={(value: string) =>
                            value === "coveredPatients"
                              ? "Funded/Covered"
                              : "Unmet Need/Risk"
                          }
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </>
            )}
          </motion.section>
        )}

        {step === 6 && (
          <motion.section
            key="step-6"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 p-8 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
          >
            <p
              className="text-center text-7xl leading-none text-black md:text-8xl"
              style={{ fontFamily: "var(--font-special-elite)" }}
            >
              ARE YOU HAPPY?
            </p>
            <div className="flex flex-wrap items-center justify-center gap-10">
              <p
                role="button"
                tabIndex={0}
                className="cursor-pointer text-2xl uppercase underline underline-offset-8 transition-transform duration-200 hover:scale-105"
                onClick={() => setStep(7)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    setStep(7);
                  }
                }}
              >
                YES
              </p>
              <p
                role="button"
                tabIndex={0}
                className="cursor-pointer text-2xl uppercase underline underline-offset-8 transition-transform duration-200 hover:scale-105"
                onClick={resetRun}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    resetRun();
                  }
                }}
              >
                TRY AGAIN
              </p>
            </div>
          </motion.section>
        )}

        {step === 7 && (
          <motion.section
            key="step-7"
            className="flex min-h-[70vh] w-full flex-col items-center justify-center gap-10 px-6 text-center md:px-16"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
          >
            <p className="max-w-5xl text-2xl leading-relaxed text-black md:text-3xl">
              {typedOutcomeNarrative}
              <span className="typewriter-cursor" />
            </p>
            <a
              href="https://project-aesclepius.streamlit.app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-4xl uppercase underline underline-offset-8 transition-transform duration-200 hover:scale-105 md:text-5xl"
            >
              VIEW FULL DATA DASHBOARD
            </a>
            <p
              role="button"
              tabIndex={0}
              className="cursor-pointer text-sm uppercase underline underline-offset-4 transition-transform duration-200 hover:scale-105"
              onClick={resetRun}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  resetRun();
                }
              }}
            >
              Play Again
            </p>
          </motion.section>
        )}
      </AnimatePresence>

      {step === 5 && simulation && !error && (
        <p
          role="button"
          tabIndex={0}
          className="fixed right-2 top-1/2 z-30 -translate-y-1/2 cursor-pointer text-lg text-black underline underline-offset-8 transition-transform duration-200 hover:scale-105 md:right-5 md:text-2xl"
          onClick={() => setStep(6)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              setStep(6);
            }
          }}
        >
          CONTINUE →
        </p>
      )}

      <footer className="fixed inset-x-0 bottom-0 z-20 border-t-4 border-black bg-black px-4 py-2 text-center text-[11px] uppercase tracking-[0.16em] text-white md:text-xs">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-center gap-4 md:gap-8">
          <a
            href="https://project-aesclepius.streamlit.app"
            target="_blank"
            rel="noopener noreferrer"
            className="underline decoration-white underline-offset-4 hover:text-zinc-300"
          >
            Detailed report (streamlit app)
          </a>
          <a
            href="https://github.com/Divyansh-Kamboj/Dubstech_Datathon"
            target="_blank"
            rel="noopener noreferrer"
            className="underline decoration-white underline-offset-4 hover:text-zinc-300"
          >
            SOURCE CODE (GitHub)
          </a>
        </div>
      </footer>
    </main>
  );
}
