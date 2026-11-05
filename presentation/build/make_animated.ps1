# Builds presentation\final_ai_project.pptx from presentation\ai project.pptx (not modified).
# Content, layout, theme and notes are unchanged; only transitions and entrance animations are set.
#
# Each slide has a list of steps. A step is revealed on a click ("click") or by itself ("auto",
# used for the opening and closing headlines). A step has one or more regions (inches: x1,y1,x2,y2);
# the shapes whose centre lies in a region appear together, and the regions of one step follow
# each other automatically (a short cascade). Titles, subtitles and header photos are not in any
# region, so they are on screen as soon as the slide appears.
# Run:  powershell -ExecutionPolicy Bypass -File presentation\build\make_animated.ps1
$ErrorActionPreference = 'Stop'
$dir = Resolve-Path (Join-Path $PSScriptRoot '..')
$src = Join-Path $dir 'ai project.pptx'
$out = Join-Path $dir 'final_ai_project.pptx'

$FADE_SMOOTHLY = 3849; $FADE_THROUGH_BLACK = 1793     # PpEntryEffect
$ANIM_FADE = 10                                        # MsoAnimEffect
$ON_CLICK = 1; $WITH_PREVIOUS = 2; $AFTER_PREVIOUS = 3 # MsoAnimTriggerType
# $false: one press shows the slide and every step then plays by itself, in the order below.
# $true: each 'click' step waits for a click.
$CLICK_TO_REVEAL = $false

$planJson = @'
{
  "1": [
    {"trigger": "auto", "regions": [[0.4, 1.4, 6.9, 2.1]]},
    {"trigger": "auto", "regions": [[0.4, 2.1, 6.9, 3.25]]},
    {"trigger": "auto", "regions": [[0.4, 3.25, 6.9, 4.3]]},
    {"trigger": "auto", "regions": [[0.4, 5.8, 13.0, 7.2]]}
  ],
  "2": [
    {"trigger": "click", "regions": [[0.5, 2.75, 4.4, 4.35]]},
    {"trigger": "click", "regions": [[4.4, 2.75, 8.4, 4.35]]},
    {"trigger": "click", "regions": [[0.5, 4.35, 4.4, 6.0]]},
    {"trigger": "click", "regions": [[4.4, 4.35, 8.4, 6.0]]},
    {"trigger": "click", "regions": [[8.6, 4.5, 13.0, 6.4], [0.5, 6.1, 8.6, 7.05]]}
  ],
  "3": [
    {"trigger": "click", "regions": [[0.5, 2.45, 2.98, 6.0]]},
    {"trigger": "click", "regions": [[2.98, 2.45, 5.43, 6.0]]},
    {"trigger": "click", "regions": [[5.43, 2.45, 7.89, 6.0]]},
    {"trigger": "click", "regions": [[7.89, 2.45, 10.34, 6.0]]},
    {"trigger": "click", "regions": [[10.34, 2.45, 12.85, 6.0]]},
    {"trigger": "click", "regions": [[0.5, 6.1, 12.85, 7.05]]}
  ],
  "4": [
    {"trigger": "click", "regions": [[0.5, 2.65, 4.6, 6.95]]},
    {"trigger": "click", "regions": [[4.6, 2.65, 8.7, 6.95]]},
    {"trigger": "click", "regions": [[8.7, 2.65, 12.85, 6.95]]}
  ],
  "5": [
    {"trigger": "click", "regions": [[0.5, 2.65, 3.6, 4.1], [3.6, 2.65, 6.7, 4.1], [6.7, 2.65, 9.8, 4.1], [9.8, 2.65, 12.85, 4.1]]},
    {"trigger": "click", "regions": [[0.5, 4.15, 3.62, 7.05], [3.62, 4.15, 6.78, 7.05]]},
    {"trigger": "click", "regions": [[6.78, 4.15, 12.85, 7.05]]}
  ],
  "6": [
    {"trigger": "click", "regions": [[0.5, 2.38, 4.75, 3.27], [0.5, 3.3, 4.75, 4.19], [0.5, 4.22, 4.75, 5.11], [0.5, 5.14, 4.75, 6.03], [0.5, 6.06, 4.75, 6.95]]},
    {"trigger": "click", "regions": [[4.85, 2.38, 7.4, 7.05]]},
    {"trigger": "click", "regions": [[7.45, 2.38, 12.85, 7.05]]}
  ],
  "7": [
    {"trigger": "click", "regions": [[0.5, 2.4, 7.6, 3.24]]},
    {"trigger": "click", "regions": [[0.5, 3.24, 7.6, 4.09], [7.9, 2.38, 12.85, 7.25]]},
    {"trigger": "click", "regions": [[0.5, 4.09, 7.6, 4.94]]},
    {"trigger": "click", "regions": [[0.5, 4.94, 7.6, 5.79]]},
    {"trigger": "click", "regions": [[0.5, 5.79, 7.6, 6.64]]}
  ],
  "8": [
    {"trigger": "click", "regions": [[0.5, 2.4, 2.92, 4.2], [2.92, 2.4, 5.34, 4.2], [5.34, 2.4, 7.8, 4.2]]},
    {"trigger": "click", "regions": [[0.5, 4.25, 2.92, 6.05], [2.92, 4.25, 5.34, 6.05], [5.34, 4.25, 7.8, 6.05]]},
    {"trigger": "click", "regions": [[7.9, 2.4, 12.85, 3.6], [7.9, 3.6, 12.85, 4.8], [7.9, 4.8, 12.85, 6.0], [7.9, 6.0, 12.85, 6.6]]}
  ],
  "9": [
    {"trigger": "click", "regions": [[0.5, 2.4, 4.62, 4.5], [4.62, 2.4, 8.76, 4.5], [8.76, 2.4, 12.85, 4.5]]},
    {"trigger": "click", "regions": [[0.5, 4.55, 4.62, 6.6], [4.62, 4.55, 8.76, 6.6]]},
    {"trigger": "click", "regions": [[8.76, 4.55, 12.85, 6.6]]}
  ],
  "10": [
    {"trigger": "auto", "regions": [[0.4, 0.3, 8.0, 2.5]]},
    {"trigger": "click", "regions": [[0.5, 3.25, 3.6, 5.3], [3.6, 3.25, 6.66, 5.3], [6.66, 3.25, 9.72, 5.3], [9.72, 3.25, 12.85, 5.3]]},
    {"trigger": "click", "regions": [[0.4, 5.6, 12.9, 6.6]]}
  ]
}
'@
$plan = $planJson | ConvertFrom-Json

if (Test-Path $out) { Remove-Item $out -Force -Confirm:$false }
$pp = New-Object -ComObject PowerPoint.Application
$deck = $pp.Presentations.Open($src, $true, $false, $false)   # read-only: the source is never saved
try {
  foreach ($slide in $deck.Slides) {
    $n = $slide.SlideIndex
    $t = $slide.SlideShowTransition
    $t.EntryEffect = if ($n -eq 10) { $FADE_THROUGH_BLACK } else { $FADE_SMOOTHLY }
    $t.Duration = [single]$(if ($n -eq 10) { 0.8 } else { 0.6 })
    $t.AdvanceOnClick = -1
    $t.AdvanceOnTime = 0

    $seq = $slide.TimeLine.MainSequence
    for ($k = $seq.Count; $k -ge 1; $k--) { $seq.Item($k).Delete() }

    # Assign each shape to the first region containing its centre.
    $steps = @($plan."$n")
    $buckets = @{}
    foreach ($s in $slide.Shapes) {
      if ($s.Type -eq 14) { continue }                      # title and kicker placeholders stay
      $cx = ($s.Left + $s.Width / 2) / 72; $cy = ($s.Top + $s.Height / 2) / 72
      $placed = $false
      for ($i = 0; $i -lt $steps.Count -and -not $placed; $i++) {
        $regs = @($steps[$i].regions)
        for ($j = 0; $j -lt $regs.Count -and -not $placed; $j++) {
          $r = $regs[$j]
          if ($cx -ge $r[0] -and $cx -lt $r[2] -and $cy -ge $r[1] -and $cy -lt $r[3]) {
            $key = "$i/$j"; if (-not $buckets.ContainsKey($key)) { $buckets[$key] = @() }
            $buckets[$key] += $s; $placed = $true
          }
        }
      }
    }

    $clicks = 0; $animated = 0
    for ($i = 0; $i -lt $steps.Count; $i++) {
      $regs = @($steps[$i].regions)
      $first = $true
      for ($j = 0; $j -lt $regs.Count; $j++) {
        $shapes = $buckets["$i/$j"]
        if (-not $shapes) { continue }
        $shapes = @($shapes | Sort-Object -Property @{Expression = { $_.Top }}, @{Expression = { $_.Left }})
        for ($k = 0; $k -lt $shapes.Count; $k++) {
          if ($k -gt 0) { $trig = $WITH_PREVIOUS }
          elseif ($first) { $trig = if ($steps[$i].trigger -eq 'click' -and $CLICK_TO_REVEAL) { $ON_CLICK } else { $AFTER_PREVIOUS } }
          else { $trig = $AFTER_PREVIOUS }
          $e = $seq.AddEffect($shapes[$k], $ANIM_FADE, 0, $trig)
          $e.Timing.Duration = 0.4
          $e.Timing.TriggerDelayTime = [single]$(if ($k -eq 0 -and -not $first) { 0.05 } else { 0 })
          $animated++
        }
        if ($first -and $steps[$i].trigger -eq 'click' -and $CLICK_TO_REVEAL) { $clicks++ }
        $first = $false
      }
    }
    $static = $slide.Shapes.Count - $animated
    "slide ${n}: $clicks click(s), $animated shapes animated, $static always visible"
  }
  $deck.SaveAs($out, 24)   # ppSaveAsOpenXMLPresentation
} finally {
  $deck.Close()
  $pp.Quit()
}
"saved $out"
