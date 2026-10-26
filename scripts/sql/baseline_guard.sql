-- Runs only on initialization of this Compose project's dedicated data volume.
CREATE TABLE public.budgetrag_baseline_guard (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    marker text NOT NULL
);
INSERT INTO public.budgetrag_baseline_guard (singleton, marker)
VALUES (true, 'budgetrag-isolated-baseline-v1');
