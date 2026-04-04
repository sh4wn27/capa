"use client";

/**
 * Re-exports framer-motion primitives tagged as client components.
 * Import from here so server components don't break.
 */
export {
  motion,
  AnimatePresence,
  useInView,
  useAnimation,
} from "framer-motion";
