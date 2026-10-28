# Adds slide-show polish to "ai project.pptx" without changing any content:
#  - a smooth fade transition between slides
#  - content fades in top to bottom, row by row, automatically (no extra clicks)
# Titles and header photos are on screen as soon as each slide appears.
# Also saves "ai project - slideshow.ppsx", which opens straight into the slide show.
# Run after build_ai_project.js:  powershell -File presentation\build\add_slideshow.ps1
$ErrorActionPreference = 'Stop'
$dir  = Resolve-Path (Join-Path $PSScriptRoot '..')
$pptx = Join-Path $dir 'ai project.pptx'
$ppsx = Join-Path $dir 'ai project - slideshow.ppsx'

$IN = 72                       # points per inch
$FADE_SMOOTHLY = 3849          # ppEffectFadeSmoothly
$ANIM_FADE = 10                # msoAnimEffectFade
$WITH_PREVIOUS = 2; $AFTER_PREVIOUS = 3
$BAND = 0.75 * $IN             # shapes starting within this height reveal together
$HEADER = 2.3 * $IN            # pictures above this line are header photos: always visible

$pp = New-Object -ComObject PowerPoint.Application
$deck = $pp.Presentations.Open($pptx, $false, $false, $false)
try {
  foreach ($slide in $deck.Slides) {
    $t = $slide.SlideShowTransition
    $t.EntryEffect = $FADE_SMOOTHLY
    $t.Duration = 0.7
    $t.AdvanceOnClick = -1

    # Remove animations from an earlier run, so the script can be re-run safely.
    $seq = $slide.TimeLine.MainSequence
    for ($k = $seq.Count; $k -ge 1; $k--) { $seq.Item($k).Delete() }

    $shapes = @()
    foreach ($s in $slide.Shapes) {
      if ($s.Type -eq 14) { continue }                         # title / kicker placeholders
      if ($s.Type -eq 13 -and $s.Top -lt $HEADER) { continue } # header or hero photo
      $shapes += $s
    }
    $shapes = $shapes | Sort-Object -Property @{Expression = { $_.Top }}, @{Expression = { $_.Left }}

    $bandTop = -1000
    foreach ($s in $shapes) {
      $newBand = $s.Top -gt ($bandTop + $BAND)
      if ($newBand) { $bandTop = $s.Top }
      $trigger = if ($newBand) { $AFTER_PREVIOUS } else { $WITH_PREVIOUS }
      $e = $seq.AddEffect($s, $ANIM_FADE, 0, $trigger)
      $e.Timing.Duration = 0.35
      $e.Timing.TriggerDelayTime = 0
    }
    "slide $($slide.SlideIndex): $($shapes.Count) shapes animated"
  }
  $deck.Save()
  $deck.SaveCopyAs($ppsx, 28)   # ppSaveAsOpenXMLShow
} finally {
  $deck.Close()
  $pp.Quit()
}
"saved $pptx"
"saved $ppsx"
