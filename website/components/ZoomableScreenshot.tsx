"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

/**
 * Screenshot wrapper that adds a click-to-zoom view on top of the
 * condensed `figure.shot` layout, so viewers can study the app showcase
 * at full size.
 */
export default function ZoomableScreenshot({
  src,
  alt,
  width,
  height,
  children,
}: {
  src: string;
  alt: string;
  width: number;
  height: number;
  children?: ReactNode;
}) {
  const [zoomed, setZoomed] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!zoomed) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setZoomed(false);
      } else if (event.key === "Tab") {
        // The dialog has one focusable control; keep focus inside it.
        event.preventDefault();
        closeRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [zoomed]);

  return (
    <>
      <figure className="shot">
        <button
          type="button"
          className="shot-zoomable"
          onClick={() => setZoomed(true)}
          aria-label={`Zoom into ${alt}`}
        >
          <img src={src} alt={alt} width={width} height={height} loading="lazy" />
        </button>
        {children && <figcaption>{children}</figcaption>}
      </figure>
      {zoomed && (
        <div
          className="zoom-overlay"
          onClick={() => setZoomed(false)}
          role="dialog"
          aria-modal="true"
          aria-label={alt}
        >
          <button
            ref={closeRef}
            type="button"
            className="zoom-close"
            onClick={(event) => {
              event.stopPropagation();
              setZoomed(false);
            }}
            aria-label="Close zoomed screenshot"
          >
            ×
          </button>
          <img
            className="zoom-view"
            src={src}
            alt={alt}
            onClick={(event) => event.stopPropagation()}
          />
          {children && <p className="zoom-view-caption">{children}</p>}
        </div>
      )}
    </>
  );
}
