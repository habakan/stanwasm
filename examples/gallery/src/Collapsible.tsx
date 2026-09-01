import { useEffect, useState, type ReactNode } from "react";

// Kept in sync by hand with the media queries in styles.css.
const NARROW = "(max-width: 720px)";

export function useIsNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia(NARROW).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(NARROW);
    const onChange = () => setNarrow(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

/**
 * Hides `children` behind a `<details>` on a narrow viewport, renders them bare
 * on a wide one. The phone scroll area is ~500px tall, so anything above the
 * plots is something to scroll past before reaching what the tab is for.
 */
export function Collapsible({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  const narrow = useIsNarrow();
  if (!narrow) return <>{children}</>;
  return (
    <details className={className ? `collapsible ${className}` : "collapsible"}>
      <summary>{label}</summary>
      <div className="collapsible-body">{children}</div>
    </details>
  );
}
