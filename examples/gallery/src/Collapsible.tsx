import { useEffect, useState, type ReactNode } from "react";

/** Matches the `max-width: 720px` breakpoint in styles.css. Kept in sync by
 *  hand rather than read from CSS, because the collapse below is a change of
 *  markup, not of styling — a phone gets fewer things open at once, and that
 *  decision has to be made in React, not in a media query. */
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
 * On a narrow viewport, hides `children` behind a `<details>` the reader can
 * open; on a wide one, renders them bare so the desktop layout is byte-for-byte
 * what it was before this component existed.
 *
 * The point is not to save pixels for their own sake. On a phone the scroll
 * area is roughly 500px tall, so anything above the plots is something the
 * reader has to scroll past before seeing the thing the tab is *for*. Supporting
 * material (the model source, the log, the long explanation) stays reachable,
 * just not in the way.
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
