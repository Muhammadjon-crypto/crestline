import json, datetime as dt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

d = json.load(open("data/replay_june2026.json"))
stage, hw = d["stage"], d["high_water_ft"]
t = [dt.datetime.fromisoformat(p["t"]) for p in stage]
y = [p["ft"] for p in stage]
peak = max(range(len(y)), key=lambda i: y[i])
crest, crest_t = y[peak], t[peak]

# when it last crossed the high-water line on the way up
cross = max(i for i in range(peak) if y[i] < hw)
rise_h = (crest_t - t[cross]).total_seconds() / 3600
print(f"crossed {hw} ft at {t[cross]:%d %b %H:%M} ({y[cross]:.2f} ft) -> "
      f"crest {crest:.2f} ft at {crest_t:%d %b %H:%M}, {rise_h:.1f} h later")

BONE, DEEP, CREST, SILT, RULE = "#F2EFE9", "#0C1D26", "#B8431A", "#5E6E76", "#DCD7CD"
have = {f.name for f in font_manager.fontManager.ttflist}
sans = next((f for f in ["Archivo","Avenir Next","Helvetica Neue","Helvetica","DejaVu Sans"] if f in have), "DejaVu Sans")
mono = next((f for f in ["IBM Plex Mono","Menlo","Monaco","DejaVu Sans Mono"] if f in have), "DejaVu Sans Mono")

fig = plt.figure(figsize=(6, 8), dpi=100, facecolor=BONE)
fig.text(.10, .915, "CrestLine", family=sans, fontsize=38, weight="bold", color=DEEP, va="center")
fig.lines.append(plt.Line2D([.10,.90],[.876,.876], transform=fig.transFigure, color=RULE, lw=1))
fig.text(.10, .852, "THREE-HOUR FLOOD FORECAST  ·  GREENS BAYOU, HOUSTON", family=mono, fontsize=8.5, color=SILT, va="center")

ax = fig.add_axes([.10, .345, .80, .50]); ax.set_facecolor(BONE)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.fill_between(t, y, min(y)-0.5, color="#EBE7DF", zorder=1)
ax.plot(t, y, color=DEEP, lw=2.4, zorder=3, solid_capstyle="round")
ax.plot(t, [v if v >= hw else float("nan") for v in y], color=CREST, lw=2.8, zorder=4, solid_capstyle="round")
ax.axhline(hw, color=CREST, lw=1, ls=(0,(5,4)), alpha=.75, zorder=2)
ax.text(t[2], hw + 0.5, f"HIGH WATER  {hw:.2f} FT", family=mono, fontsize=7.5, color=CREST, va="bottom")
ax.plot([crest_t],[crest], "o", ms=5.5, color=CREST, zorder=5)
ax.annotate(f"{crest:.2f} ft", (crest_t, crest), xytext=(-6, 12), textcoords="offset points",
            family=sans, fontsize=15, weight="bold", color=DEEP, ha="right")
ax.set_ylim(min(y)-0.5, crest*1.16)

fig.text(.10, .252, f"{crest_t.day} {crest_t:%B %Y}".upper(), family=sans,
         fontsize=27, weight="bold", color=DEEP, va="center")
fig.text(.10, .196, f"The bayou passed high water and crested at {crest:.2f} ft\n{rise_h:.0f} hours later. "
                    "CrestLine sees it coming three hours out.",
         family=sans, fontsize=12.5, color="#3D4C54", va="top", linespacing=1.5)
fig.lines.append(plt.Line2D([.10,.90],[.112,.112], transform=fig.transFigure, color=RULE, lw=1))
fig.text(.10, .078, "USGS 08076700  ·  CONGRESSIONAL APP CHALLENGE 2026", family=mono, fontsize=8.5, color=SILT, va="center")

fig.savefig("cover_crestline.jpg", dpi=100, facecolor=BONE, pil_kwargs={"quality": 94})
print("wrote cover_crestline.jpg (600x800)")
