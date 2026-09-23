import os
import sys
import math
import requests
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from jinja2 import Environment, FileSystemLoader

# Every GitHub/site request gets a timeout so a stalled connection fails the
# run instead of hanging the job until Actions kills it hours later.
API_TIMEOUT = 30

# Commits are bucketed into days in this timezone (by authored time), so an
# evening commit lands on the day it was actually written rather than on
# the next UTC day.
PROFILE_TZ = ZoneInfo(os.environ.get("PROFILE_TZ", "America/Denver"))


def format_size(n):
    """Bytes -> human-readable string, e.g. 1234567 -> '1.2 MB'."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"

# Repos to leave out of the Languages card entirely -- e.g. pulled-in
# collections/amalgamations of other projects that aren't really "your"
# code. Doesn't affect stars/streak/contributions, just the language mix.
LANGUAGE_EXCLUDE_REPOS = {
    "clankinator",
    "HannahMontanaLinux",
}

# Languages that are usually generated/config rather than hand-written,
# so they shouldn't count toward "what do I code in".
LANGUAGE_EXCLUDE_LANGS = {
    "XML",
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "HCL",
    "Roff",
    "QML",
    "CMake",
    "CSS",
    "Makefile",
    "Dockerfile",
    "Linker Script",
    "Markdown",
    "HTML",
}


CAL_WEEKS = 48
CAL_PITCH = 13
CAL_CELL = 11


def _graphql(headers, query, variables):
    resp = requests.post("https://api.github.com/graphql",
                         json={"query": query, "variables": variables},
                         headers=headers, timeout=API_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error: {resp.status_code} - {resp.text}")
    return resp.json()


def _walk_repo_commits(headers, owner, name, author_id, since_iso):
    """Authored dates of `author_id`'s commits on one repo's default branch.

    This deliberately avoids GraphQL's `contributionsCollection` field: that
    field's totals/streaks are gated by the "Include private contributions
    on my profile" setting, and are zeroed out for *everyone* -- including
    the account owner's own token -- when that setting is off. Walking each
    repo's own commit history isn't subject to that toggle: it's just repo
    data the PAT already has read access to (same access the Languages card
    relies on), so this keeps the profile's public privacy setting exactly
    as-is while still getting real numbers for this script.
    """
    query = """
    query($owner: String!, $name: String!, $since: GitTimestamp!, $author: ID!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(since: $since, author: {id: $author}, first: 100, after: $cursor) {
                pageInfo { hasNextPage endCursor }
                nodes { authoredDate }
              }
            }
          }
        }
      }
    }
    """
    dates = []
    cursor = None
    while True:
        variables = {"owner": owner, "name": name, "since": since_iso,
                     "author": author_id, "cursor": cursor}
        data = _graphql(headers, query, variables)
        if "errors" in data:
            # A repo that vanished between listing and walking is fine to
            # skip; anything else (rate limit, timeout, permissions) would
            # silently shrink the totals, so fail the run instead.
            if all(e.get("type") == "NOT_FOUND" for e in data["errors"]):
                return dates
            raise RuntimeError(f"GitHub GraphQL errors on {name}: {data['errors']}")
        repo_data = data["data"]["repository"]
        ref = repo_data.get("defaultBranchRef") if repo_data else None
        target = ref.get("target") if ref else None
        if not target:
            # Empty repo / no default branch.
            return dates
        history = target["history"]
        dates.extend(node["authoredDate"] for node in history["nodes"])
        if not history["pageInfo"]["hasNextPage"]:
            return dates
        cursor = history["pageInfo"]["endCursor"]


def _walk_commit_days(headers, owner, repo_names, author_id, since_iso):
    """Tally `author_id`'s commits per local day, across `repo_names`."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        per_repo = pool.map(
            lambda name: _walk_repo_commits(headers, owner, name, author_id, since_iso),
            repo_names)
        daily_counts = {}
        for dates in per_repo:
            for iso in dates:
                day = datetime.fromisoformat(iso).astimezone(PROFILE_TZ).date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
    return daily_counts


def _calendar_start(end_date):
    """Sunday that begins the first of the CAL_WEEKS columns ending at `end_date`."""
    this_sunday = end_date - timedelta(days=end_date.isoweekday() % 7)
    return this_sunday - timedelta(weeks=CAL_WEEKS - 1)


def _build_weeks(daily_counts, start_date, end_date):
    """Sunday-anchored weeks of contributionDays, same shape the template expects.

    Days after `end_date` are left out, so the last column stops at today
    instead of showing empty cells for days that haven't happened yet.
    """
    weeks = []
    cur = start_date
    while cur <= end_date:
        week_days = []
        for i in range(7):
            d = cur + timedelta(days=i)
            if d > end_date:
                break
            date_str = d.isoformat()
            week_days.append({"contributionCount": daily_counts.get(date_str, 0), "date": date_str})
        weeks.append({"contributionDays": week_days})
        cur += timedelta(days=7)
    return weeks


def _compute_streaks(daily_counts, start_date, end_date):
    counts = []
    d = start_date
    while d <= end_date:
        counts.append(daily_counts.get(d.isoformat(), 0))
        d += timedelta(days=1)
    current_streak = 0
    for i, c in enumerate(reversed(counts)):
        if c > 0:
            current_streak += 1
        elif i == 0:
            # end_date is "today"; a 0 there just means today isn't over
            # yet (the cron runs at 00:00 UTC, i.e. evening locally), not
            # a broken streak.
            continue
        else:
            break
    longest_streak = run = 0
    for c in counts:
        if c > 0:
            run += 1
            longest_streak = max(longest_streak, run)
        else:
            run = 0
    return current_streak, longest_streak


def _fetch_repos(headers):
    """The viewer's node id, plus all their owned, non-fork repos + top languages."""
    query = """
    query($cursor: String) {
      viewer {
        id
        repositories(first: 100, ownerAffiliations: OWNER, isFork: false, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes {
            name
            languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
              edges { size node { name } }
            }
          }
        }
      }
    }
    """
    repos = []
    cursor = None
    while True:
        data = _graphql(headers, query, {"cursor": cursor})
        if "errors" in data:
            raise RuntimeError(f"GitHub GraphQL errors: {data['errors']}")
        viewer = data["data"]["viewer"]
        repositories = viewer["repositories"]
        repos.extend(repositories["nodes"])
        if repositories["pageInfo"]["hasNextPage"]:
            cursor = repositories["pageInfo"]["endCursor"]
        else:
            break
    return viewer["id"], repos


def contribution_stats(daily_counts, end_date):
    """Calendar weeks, total and streaks, all over the same displayed window."""
    start_date = _calendar_start(end_date)
    total = 0
    d = start_date
    while d <= end_date:
        total += daily_counts.get(d.isoformat(), 0)
        d += timedelta(days=1)
    current_streak, longest_streak = _compute_streaks(daily_counts, start_date, end_date)
    return {
        "total_contribs": total,
        "weeks": _build_weeks(daily_counts, start_date, end_date),
        "current_streak": current_streak,
        "longest_streak": longest_streak,
    }


def get_stats(token, username):
    headers = {"Authorization": f"Bearer {token}"}
    viewer_id, repos = _fetch_repos(headers)

    # languages
    langs = {}
    for repo in repos:
        if repo["name"] in LANGUAGE_EXCLUDE_REPOS:
            continue
        for edge in repo["languages"]["edges"]:
            name = edge["node"]["name"]
            if name in LANGUAGE_EXCLUDE_LANGS:
                continue
            size = edge["size"]
            langs[name] = langs.get(name, 0) + size
    sorted_langs = sorted(langs.items(), key=lambda x: x[1], reverse=True)[:6]

    # True bar chart: bar length is each language's size relative to the
    # largest one shown (not a percentage of the whole), so it reads as
    # "how much bigger is JS than C" rather than "share of a pie".
    max_size = sorted_langs[0][1] if sorted_langs else 1
    processed_langs = [
        {
            "name": name,
            "size_label": format_size(size),
            "bar_pct": (size / max_size * 100) if max_size else 0,
        }
        for name, size in sorted_langs
    ]

    # contributions + streak
    # Built from raw per-repo commit history rather than
    # contributionsCollection -- see _walk_repo_commits for why. The fetch
    # starts a day early so commits from the window's first local day
    # aren't cut off by the UTC offset.
    end_date = datetime.now(PROFILE_TZ).date()
    since_iso = (_calendar_start(end_date) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    repo_names = [r["name"] for r in repos]
    daily_counts = _walk_commit_days(headers, username, repo_names, viewer_id, since_iso)

    # open-source PRs
    # contributionsCollection-based counts (repositoriesContributedTo etc.)
    # come back empty on this account for reasons unrelated to this script
    # (see git history), so this uses the search API instead, which counts
    # PRs authored on repos not owned by this user directly. Only merged
    # PRs on public repos count -- open, closed-unmerged, or private-repo
    # ones don't.
    search_resp = requests.get(
        "https://api.github.com/search/issues",
        params={"q": f"type:pr is:merged is:public author:{username} -user:{username}"},
        headers=headers, timeout=API_TIMEOUT)
    if search_resp.status_code != 200:
        raise RuntimeError(f"GitHub search API error: {search_resp.status_code} - {search_resp.text}")
    search = search_resp.json()
    if search.get("incomplete_results"):
        # Search timed out server-side; total_count would be an undercount.
        raise RuntimeError("GitHub search returned incomplete results for PR count")
    oss_prs = search["total_count"]

    return {
        "langs": processed_langs,
        "oss_prs": oss_prs,
        **contribution_stats(daily_counts, end_date),
    }


def generate_stars(seed=7):
    """A scattered, twinkling star field for the sky.

    Each star gets its own twinkle period and phase so the sky shimmers
    unevenly instead of whole groups pulsing in lockstep; some stars hold
    steady. Stars cluster along a diagonal Milky Way band and thin out
    toward the horizon. Seeded, so the sky is identical run to run and
    stats.svg only changes when the stats do.
    """
    rng = random.Random(seed)
    stars = []

    def add(x, y, r, brightness):
        star = {"x": round(x, 1), "y": round(y, 1), "r": round(r, 2),
                "opacity": round(brightness, 2), "dur": None, "delay": None}
        if rng.random() < 0.75:
            dur = rng.uniform(2.5, 7.5)
            star["dur"] = round(dur, 2)
            # Negative delay starts each star partway through its cycle.
            star["delay"] = round(-rng.uniform(0, dur), 2)
        stars.append(star)

    # Milky Way band: centred on (310, 190), tilted -22 degrees, matching
    # the soft glow ellipse drawn behind it.
    angle = math.radians(-22)
    for _ in range(150):
        t = rng.uniform(-470, 470)
        off = rng.gauss(0, 30)
        x = 310 + t * math.cos(angle) - off * math.sin(angle)
        y = 190 + t * math.sin(angle) + off * math.cos(angle)
        if 0 <= x <= 800 and 0 <= y <= 330:
            add(x, y, rng.uniform(0.4, 1.1), rng.uniform(0.4, 0.9))

    # Field stars: denser overhead, thinning toward the horizon glow.
    for _ in range(170):
        x = rng.uniform(0, 800)
        y = 330 * rng.random() ** 1.6
        add(x, y, rng.uniform(0.4, 1.0), rng.uniform(0.3, 0.75))

    # A few brighter foreground stars.
    for _ in range(12):
        add(rng.uniform(10, 790), rng.uniform(5, 260), rng.uniform(1.3, 1.9), rng.uniform(0.8, 1.0))

    return stars


def generate_svg(stats, site):
    # Classic GitHub-style contribution calendar: squares, weekday/month labels,
    # ~11 months of history so it actually reads as a full activity chart.
    calendar_dots = []
    month_labels = []
    recent_weeks = stats["weeks"][-CAL_WEEKS:]
    prev_month = None

    for col_idx, week in enumerate(recent_weeks):
        for day in week["contributionDays"]:
            # Parse date to get day of week (0=Sunday, 1=Monday, ... 6=Saturday)
            date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
            # isoweekday() is 1(Mon)-7(Sun). We want 0(Sun)-6(Sat)
            row_idx = date_obj.isoweekday() % 7

            c = day["contributionCount"]
            if c == 0:
                level = 0
            elif c <= 2:
                level = 1
            elif c <= 5:
                level = 2
            elif c <= 8:
                level = 3
            else:
                level = 4

            calendar_dots.append({
                "x": col_idx * CAL_PITCH,
                "y": row_idx * CAL_PITCH,
                "count": c,
                "level": level,
                "date": day["date"],
                "delay": round((col_idx * 7 + row_idx) * 0.004, 3),
            })

            if row_idx == 0:
                month = date_obj.strftime("%b")
                if month != prev_month:
                    x = col_idx * CAL_PITCH
                    # A month that starts within a column or two of the last label
                    # would print on top of it, so skip the label (not the month).
                    if not month_labels or x - month_labels[-1]["x"] >= 3 * CAL_PITCH:
                        month_labels.append({"x": x, "label": month})
                    prev_month = month

    env = Environment(loader=FileSystemLoader('.'), autoescape=True)
    env.filters["commas"] = lambda n: f"{n:,}"
    template = env.get_template('template.svg.j2')

    svg_content = template.render(
        stats=stats,
        calendar_dots=calendar_dots,
        month_labels=month_labels,
        cal_cell=CAL_CELL,
        cal_pitch=CAL_PITCH,
        stars=generate_stars(),
        site=site,
    )
    return svg_content


def get_site_widgets():
    """Small, best-effort widgets sourced from emma-stensland.com itself.

    Every network call here is wrapped and swallowed on failure -- the
    personal site has its own uptime, independent of GitHub, and a hiccup
    there should never break this SVG's generation. Card renders with
    whatever subset came back (falls back to just the plain link if both
    fail).

    The bot-catch count deliberately hits the *ungated* tier of
    /api/bot-stats (no Turnstile token sent) -- same opaque aggregate,
    no-mechanism-vocabulary shape the site's own homepage displays to
    anonymous visitors. See bot-stats.js's own comments for why that
    shape is safe to expose.

    GitHub Actions' outbound IPs read as automated/datacenter traffic to
    the site's own bot defenses, so requests can get scored as a bot and
    blocked/tarpitted before ever reaching the handler. PROFILE_WIDGET_KEY
    (a GitHub Actions secret) is sent as X-Profile-Widget-Key so the site
    side can allowlist this one caller by shared secret rather than by IP
    (GH Actions ranges are broad and rotate) -- requires a matching check
    added on the site side; this header is a no-op until that exists.
    """
    site = {"guestbook_count": None, "bot_count": None, "poison_size": None}
    widget_key = os.environ.get("PROFILE_WIDGET_KEY")
    headers = {"X-Profile-Widget-Key": widget_key} if widget_key else {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        guestbook_future = pool.submit(_fetch_json, "guestbook", "https://emma-stensland.com/api/guestbook", headers)
        bot_stats_future = pool.submit(_fetch_json, "bot-stats", "https://emma-stensland.com/api/bot-stats", headers)
        guestbook_data = guestbook_future.result()
        bot_stats_data = bot_stats_future.result()

    # Only trust bodies of the expected shape; anything else just drops
    # that line from the card.
    if isinstance(guestbook_data, dict) and isinstance(guestbook_data.get("entries"), list):
        site["guestbook_count"] = len(guestbook_data["entries"])

    if isinstance(bot_stats_data, dict):
        count = bot_stats_data.get("count")
        if isinstance(count, (int, float)):
            site["bot_count"] = int(count)
        poison_bytes = bot_stats_data.get("poisonBytes")
        if isinstance(poison_bytes, (int, float)):
            site["poison_size"] = format_size(poison_bytes)

    return site


def _fetch_json(label, url, headers):
    """GET `url` and return its parsed JSON body, or None on any failure."""
    try:
        r = requests.get(url, headers=headers, timeout=8)
        print(f"[site-widgets] {label}: HTTP {r.status_code}")
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[site-widgets] {label}: FAILED ({type(e).__name__}: {e})")
    return None


if __name__ == "__main__":
    token = os.environ.get("GH_TOKEN")
    username = os.environ.get("GH_USERNAME", "stenslae")
    stats = None

    if token:
        # No fallback here on purpose: if the API call fails, let the
        # workflow fail loudly (non-zero exit) rather than silently
        # committing mock data as if it were real.
        print(f"Fetching stats for {username} from GitHub GraphQL API...")
        stats = get_stats(token, username)
        site = get_site_widgets()
    elif os.environ.get("GITHUB_ACTIONS") == "true":
        # In CI an empty token means the PROFILE_PAT secret is missing or
        # unreadable; mock data here would get committed as real stats.
        sys.exit("GH_TOKEN is empty in CI -- is the PROFILE_PAT secret set? Refusing to commit mock data.")
    else:
        print("Running with mock data...")
        # Clean, realistic mock data. Doesn't touch the network at all,
        # including the personal site.
        mock_rand = random.Random(42)
        end_date = datetime.now(PROFILE_TZ).date()
        daily_counts = {}
        d = _calendar_start(end_date)
        while d <= end_date:
            if mock_rand.random() > 0.45:
                daily_counts[d.isoformat()] = mock_rand.randint(1, 14)
            d += timedelta(days=1)
        stats = {
            "langs": [
                {"name": "JavaScript", "size_label": "10.2 MB", "bar_pct": 100.0},
                {"name": "C", "size_label": "9.9 MB", "bar_pct": 97.1},
                {"name": "Assembly", "size_label": "897.0 KB", "bar_pct": 8.6},
                {"name": "VHDL", "size_label": "365.0 KB", "bar_pct": 3.5},
                {"name": "C++", "size_label": "182.0 KB", "bar_pct": 1.7},
                {"name": "MATLAB", "size_label": "134.0 KB", "bar_pct": 1.3},
            ],
            "oss_prs": 1023,
            **contribution_stats(daily_counts, end_date),
        }
        site = {"guestbook_count": 4, "bot_count": 76441423, "poison_size": "637.38 GB"}

    svg = generate_svg(stats, site)
    with open("stats.svg", "w") as f:
        f.write(svg)
    print("Generated stats.svg successfully!")
