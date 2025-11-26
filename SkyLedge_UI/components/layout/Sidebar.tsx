"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

function cn(...cls: Array<string | false | null | undefined>) {
  return cls.filter(Boolean).join(" ");
}

type Item = {
  href?: string;
  label: string;
  icon?: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  badge?: string;
};

export default function Sidebar({
  title,
  subtitle,
  items,
  onNavigate,
  className,
}: {
  title: string;
  subtitle?: string;
  items: Item[];
  onNavigate?: () => void;
  className?: string;
}) {
  const pathname = usePathname();
  const [hash, setHash] = useState<string>("");

  useEffect(() => {
    const updateHash = () => setHash(typeof window !== "undefined" ? window.location.hash : "");
    updateHash();
    window.addEventListener("hashchange", updateHash);
    return () => window.removeEventListener("hashchange", updateHash);
  }, []);
  return (
    <aside className={cn("w-64 h-full bg-card border-r border-border p-4", className)}>
      <div className="px-2 pb-4">
        <h1 className="text-lg font-bold">{title}</h1>
        {subtitle ? <p className="text-xs text-muted-foreground">{subtitle}</p> : null}
      </div>

      <nav className="space-y-1">
        {items.map((item, idx) => {
          // Support links with hashes (e.g., "/developer#upload").
          const [itemPath, itemHash] = item.href ? item.href.split("#") : ["", ""];
          // Strict equality for path to avoid base route highlighting subroutes like /user on /user/upload-trip
          const matchesPathStrict = item.href ? pathname === itemPath : false;
          const matchesHashStrict = itemHash ? hash === `#${itemHash}` : true;
          // If item has a hash, require both exact path and exact hash. If no hash, require exact path only.
          const isActive = item.href ? (itemHash ? (matchesPathStrict && matchesHashStrict) : matchesPathStrict) : false;
          const Icon = item.icon;
          if (!item.href) {
            return (
              <div
                key={`label-${item.label}-${idx}`}
                className="px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground/80 select-none"
              >
                {item.label}
              </div>
            );
          }
          return (
            <Link
              key={`${item.href}-${idx}`}
              href={item.href}
              onClick={(e) => {
                // If navigating to a hash on the same page, update hash and notify listeners immediately
                const [navPath, navHash] = (item.href || "").split("#");
                const isSamePath = pathname === navPath || (navPath === "" && !!pathname);
                if (navHash && isSamePath) {
                  e.preventDefault();
                  const next = `${navPath}#${navHash}`;
                  if (typeof window !== "undefined") {
                    history.pushState(null, "", next);
                    window.dispatchEvent(new HashChangeEvent("hashchange"));
                  }
                }
                onNavigate?.();
              }}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                isActive ? "bg-primary text-primary-foreground" : "hover:bg-muted"
              )}
            >
              {Icon ? <Icon className="h-4 w-4" /> : null}
              <span className="flex-1">{item.label}</span>
              {item.badge ? (
                <span
                  className={cn(
                    "text-[10px] px-1.5 py-0.5 rounded",
                    isActive ? "bg-primary-foreground text-primary" : "bg-muted text-foreground"
                  )}
                >
                  {item.badge}
                </span>
              ) : null}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
