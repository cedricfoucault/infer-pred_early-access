import {TWOPI, DEG_TO_RAD, mod, subtractAngle} from './utils/num.js';
import {BET_WIDTHS_RAD, SCORE_GAIN_TABLE, INITIAL_VALUES,
 N_POINTS_PER_COIN, nCoinsWithPoints} from './taskParams.js';

//
// UI (User interface) constants
//

export const COLORS = {
  white: "white",
  black: "black",
  background: "#E5E5E5",
  defaultTextColor: "black",
  defaultStrokeColor: "black",
  ringFill: "rgb(128, 128, 128)",
  ringFillHit: "rgb(179, 179, 179)",
  laser: "rgb(255 221 51 / 90%)",
  sliderFill: {
    precise: "#FFFFFF",
    imprecise: "rgb(239 239 239)",
    preciseCatching: "rgb(255 229.5 102)",
    // impreciseCatching: "rgb(247 230 145)",
    impreciseCatching: "hsl(50 60 90)",
  },
  scoreBar: {
    fill: "#808080",
    gainFill: "hsl(50 100 75)",
    lossFill: "hsl(0 0 80)"
  },
  target: "black",
};

export const SIZES = {
  radius: {
    ringInnerCircle: 65,
    ringOuterCircle: 75,
    laser: 70,
    source: 5,
    targetLine: 70,
    targetTip: 4,
  },
  width: {
    scoreBar: 12,
    totalScoreDisplay: 130,
  },
  height: {
    scoreBar: 100,
  },
  lineWidth: {
    ring: 2,
    source: 2,
    slider: 2,
    laser: 3,
    scoreBar: 1,
    targetLine: 2,
  }
};

export const LAYOUT = {
  scoreBarCenterXToRingCenterX: -150,
  scoreBarBottomToRingCenterY: -75,
  scoreBarTopLabelBaselineToBarTop: 6,
  scoreBarBottomToBottomLabelBaseline: 14,
  scoreBarRightToGainLabelLeft: 6,
  scoreBarBottomLabelBaselineToTotalScoreLabelBaseline: 22,
  totalScoreLabelBaselineToCoinsLabelTop: 14,
  practiceTextBaseLineToRingCenterY: 40 + SIZES.radius.ringOuterCircle,
}

export const FONTS = {
  default: "Arial",
  score: "Courier New",
  emoji: "emoji",
  size: {
    default: 12,
    scoreBarLabels: 12,
    gainedPoints: 14,
    gainedCoin: 14,
    totalScore: 12,
    totalCoins: 8,
    practiceText: 14,
  },
  lineHeightMultiplier: 1.5,
};

export const EMOJI = {
  coin: "🟡",
  crossmark: "❌"
};

export const TEXT = {
  scoreBar: {
    topLabel: N_POINTS_PER_COIN.toString(),
    bottomLabel: "Pt",
  },
  totalScoreLabel(score) {
    return `Total: ${score} pts`;
  },
  coinsLabel(nCoins) {
    return Array(nCoins).fill(EMOJI.coin).join(" ");
  }
}

const DRAW_ANGLE_OFFSET = -Math.PI/2;

//
// UI utility functions
//

function drawText(ctx, text, x, y, fontSize = FONTS.size.default,
  textAlign = "center",
  textBaseline = "alphabetic",
  color = COLORS.defaultTextColor,
  bold = false,
  font = FONTS.default) {
  ctx.font = (bold ? "bold " : "") + String(fontSize) + "px " + font;
  ctx.textAlign = textAlign;
  ctx.textBaseline = textBaseline;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function measureText(ctx, text, fontSize,
  textAlign = "center",
  textBaseline = "alphabetic",
  bold = false,
  font = FONTS.default) {
  ctx.font = getFont(fontSize, bold, font);
  ctx.textAlign = textAlign;
  ctx.textBaseline = textBaseline;
  return ctx.measureText(text);
}

function getFont(fontSize = FONTS.size.default, bold = false, font = FONTS.default) {
  return (bold ? "bold " : "") + String(fontSize) + "px " + font;
}

// Calculate height for the text displayed on a single line
function getTextHeight(ctx, text, fontSize) {
  const previousTextBaseline = ctx.textBaseline;
  const previousFont = ctx.font;

  ctx.textBaseline = 'bottom';
  ctx.font = getFont(fontSize, bold, font);
  const textMetrics = ctx.measureText(text);
  const height = textMetrics.actualBoundingBoxAscent;

  ctx.textBaseline = previousTextBaseline;
  ctx.font = previousFont;

  return height;
}

//
// UI Classes
//

export class Display {
  constructor() {
    this._hidden = false;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(value) {
    if (value != this._hidden) {
      this._hidden = value;
      setNeedsRedraw();
    }
  }

  setHiddenWithoutRedraw(value = true) {
    this._hidden = value;
  }
}

export class TaskDisplay extends Display {
  constructor(centerX, centerY, canvasWidth = 640) {
    super();

    this.source = new Source(centerX, centerY);
    this.ring = new Ring(centerX, centerY);
    this.slider = new Slider(centerX, centerY, 0);
    this.laser = new Laser(centerX, centerY, 0, true);
    this.scoreDisplay = new ScoreDisplay(null,
      centerX - LAYOUT.scoreBarCenterXToRingCenterX,
      centerY - LAYOUT.scoreBarBottomToRingCenterY);
    this.targetLocation = new TargetLocation(centerX, centerY, 0, true);
    this._practiceText = null;
    this.canvasWidth = canvasWidth;
  }

  draw(ctx) {
    this.scoreDisplay.draw(ctx);
    this.source.draw(ctx);
    this.ring.draw(ctx);
    this.slider.draw(ctx);
    this.laser.draw(ctx);
    this.targetLocation.draw(ctx);
    if (this.practiceText) {
      const bottomBaselineY = this.ring.centerY - LAYOUT.practiceTextBaseLineToRingCenterY;
      const label = new MultilineTextDisplay(ctx, this.ring.centerX, bottomBaselineY,
        this.practiceText, this.canvasWidth, "center", "alphabetic",
        FONTS.default, FONTS.size.practiceText, COLORS.defaultTextColor,
        FONTS.lineHeightMultiplier);
      if (label.textLines && (label.textLines.length > 1)) {
        label.topAnchorY = bottomBaselineY - (label.textLines.length - 1) * label.lineHeight;
      }
      label.draw(ctx);
    }
  }

  showReadyCue() {
    this.source.active = true;
  }

  showLaser(location) {
    this.source.active = true;
    this.laser.location = location;
    this.laser.hidden = false;
  }

  hideLaser() {
    this.laser.hidden = true;
    this.source.active = false;
  }

  showWinOutcome(didCatch) {
    this.slider.catching = didCatch;
    this.ring.hit = !didCatch;
    this.showGain(didCatch);
  }

  hideWinOutcome() {
    this.slider.catching = false;
    this.ring.hit = false;
    this.hideGain();
  }

  showGain(didCatch) {
    const k1 = didCatch ? "win" : "lose";
    const k2 = (this.slider.widthIdx > 0 ? "imprecise" : "precise");
    const gain = SCORE_GAIN_TABLE[k1][k2];
    if (gain != 0) {
      this.scoreDisplay.addToScore(gain);
      this.scoreDisplay.gainVisible = true;
    }
  }

  hideGain() {
    this.scoreDisplay.gainVisible = false;
  }

  showTarget(location) {
    this.targetLocation.location = location;
    this.targetLocation.hidden = false;
  }

  hideTarget() {
    this.targetLocation.hidden = true;
  }

  get practiceText() {
    return this._practiceText;
  }

  set practiceText(text) {
    if (text != this._practiceText) {
      this._practiceText = text;
      setNeedsRedraw();
    }
  }
}

class MultilineTextDisplay extends Display {
  constructor(ctx, anchorX, topAnchorY, text, width,
    textAlign = "center", textBaseline = "alphabetic",
    font = FONTS.default, fontSize = FONTS.size.default,
    color = COLORS.defaultTextColor,
    lineHeightMultiplier = FONTS.lineHeightMultiplier) {
    super();
    this.anchorX = anchorX;
    this.topAnchorY = topAnchorY;
    this.text = text;
    this.width = width;
    this.fontSize = fontSize;
    this.font = font;
    // this.lineHeightMultiplier = lineHeightMultiplier;
    this.textAlign = textAlign;
    this.color = color;
    this.lineHeight = (lineHeightMultiplier ?
      Math.ceil(fontSize * lineHeightMultiplier)
      : getTextHeight(ctx, this.text, this.fontSize));
    this.textLines = this.calculateTextLines(ctx);
    this.topToBottomAnchor = this.lineHeight * (this.textLines.length - 1);
  }

  get bottomAnchorY() {
    return this.topAnchorY + this.topToBottomAnchor;
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.font = getFont(this.fontSize, false, this.font);
    ctx.textAlign = this.textAlign;
    ctx.textBaseline = this.textBaseline;
    ctx.fillStyle = this.color;

    // print all lines of text
    let txtY = this.topAnchorY;
    this.textLines.forEach(txtline => {
      txtline = txtline.trim();
      ctx.fillText(txtline, this.anchorX, txtY);
      txtY += this.lineHeight;
    });
  }

  // adapted from https://github.com/geongeorge/Canvas-Txt/blob/master/src/index.js
  calculateTextLines(ctx) {
    const width = this.width;
    const lineHeight = this.lineHeight;

    ctx.font = getFont(this.fontSize, false, this.font);
    ctx.textAlign = this.textAlign;
    ctx.textBaseline = this.textBaseline;

    // added one-line only auto linebreak feature
    let textarray = [];
    let temptextarray = this.text.split('\n');
    temptextarray.forEach(txtt => {
      let textwidth = ctx.measureText(txtt).width;
      if (textwidth <= width) {
        textarray.push(txtt)
      } else {
        let temptext = txtt;
        let linelen = width;
        let textlen;
        let textpixlen;
        let texttoprint;
        textwidth = ctx.measureText(temptext).width;
        while (textwidth > linelen) {
          textlen = 0;
          textpixlen = 0;
          texttoprint = '';
          while (textpixlen < linelen) {
            textlen++;
            texttoprint = temptext.substr(0, textlen);
            textpixlen = ctx.measureText(temptext.substr(0, textlen)).width;
          }
          // Remove last character that was out of the box
          textlen--;
          texttoprint = texttoprint.substr(0, textlen);
          //if statement ensures a new line only happens at a space, and not amidst a word
          const backup = textlen;
          if (temptext.substr(textlen, 1) != ' ') {
            while (temptext.substr(textlen, 1) != ' ' && textlen != 0) {
              textlen--;
            }
            if (textlen == 0) {
              textlen = backup;
            }
            texttoprint = temptext.substr(0, textlen);
          }

          temptext = temptext.substr(textlen);
          textwidth = ctx.measureText(temptext).width;
          textarray.push(texttoprint);
        }
        if (textwidth > 0) {
          textarray.push(temptext);
        }
      }
      // end foreach temptextarray
    })

    return textarray;
  }
}

class MovableDisplay extends Display {
  constructor(location) {
    super();
    this._location = location;
  }

  get location() {
    return this._location;
  }

  set location(value) {
    if (this._location != value) {
      this._location = value;
      setNeedsRedraw();
    }
  }
}

class Source extends Display {
  constructor(centerX, centerY,
    hidden = false, radius = SIZES.radius.source,
    fillColorInactive = COLORS.background,
    fillColorActive = COLORS.laser) {
    super();
    this.centerX = centerX;
    this.centerY = centerY;
    this.radius = radius;
    this.fillColorInactive = fillColorInactive;
    this.fillColorActive = fillColorActive;
    this._active = false;
    this.setHiddenWithoutRedraw(hidden);
  }

  get active() {
    return this._active;
  }

  set active(value) {
    if (this._active != value) {
      this._active = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.lineWidth = SIZES.lineWidth.source;
    ctx.strokeStyle = COLORS.defaultStrokeColor;
    ctx.fillStyle = this.active ? this.fillColorActive : this.fillColorInactive;
    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, this.radius, 0, TWOPI);
    ctx.stroke();
    ctx.fill();
    ctx.closePath();
  }
}

class Ring extends Display {
  constructor(centerX, centerY,
    hidden = false, innerRadius = SIZES.radius.ringInnerCircle,
    outerRadius = SIZES.radius.ringOuterCircle,
    ) {
    super();
    this.centerX = centerX;
    this.centerY = centerY;
    this.innerRadius = innerRadius;
    this.outerRadius = outerRadius;
    this.setHiddenWithoutRedraw(hidden);
    this._hit = false;
  }

  get fillColor() {
    return this.hit ? COLORS.ringFillHit : COLORS.ringFill;
  }

  get hit() {
    return this._hit;
  }

  set hit(value) {
    if (this._hit != value) {
      this._hit = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.lineWidth = SIZES.lineWidth.ring;
    ctx.strokeStyle = COLORS.defaultStrokeColor;
    ctx.fillStyle = this.fillColor;

    
    ctx.beginPath();
    // Outer circle
    ctx.arc(this.centerX, this.centerY, this.outerRadius, 0, TWOPI);
    // Inner circle
    ctx.arc(this.centerX, this.centerY, this.innerRadius, 0, TWOPI, true);
    ctx.stroke();
    ctx.fill();
    ctx.closePath();
  }

  get midRadius() {
    return (this.innerRadius + this.outerRadius) / 2;
  }
}

class Slider extends MovableDisplay {
  constructor(arcCenterX, arcCenterY, location = INITIAL_VALUES.betLoc,
    widthIdx = INITIAL_VALUES.betWidIdx,
    widths = BET_WIDTHS_RAD,
    hidden = false, innerRadius = SIZES.radius.ringInnerCircle,
    outerRadius = SIZES.radius.ringOuterCircle,
    ) {
    super(location);
    this._widthIdx = widthIdx;
    this._widths = widths;
    this._catching = false;
    this.arcCenterX = arcCenterX;
    this.arcCenterY = arcCenterY;
    this.innerRadius = innerRadius;
    this.outerRadius = outerRadius;
    this.setHiddenWithoutRedraw(hidden);
  }

  get location() {
    return this._location;
  }

  set location(value) {
    if (this._location != value) {
      this._location = value;
      setNeedsRedraw();
    }
  }

  get isCatching() {
    return this._catching;
  }

  set catching(value) {
    if (this._catching != value) {
      this._catching = value;
      setNeedsRedraw();
    }
  }

  get widthIdx() {
    return this._widthIdx;
  }

  set widthIdx(value) {
    if (this._widthIdx != value) {
      this._widthIdx = value;
      setNeedsRedraw();
    }
  }

  get width() {
    return this._widths[this._widthIdx];
  }

  get fillColor() {
    if (this.widthIdx > 0) {
      return this.isCatching ? COLORS.sliderFill.impreciseCatching : COLORS.sliderFill.imprecise;
    } else {
      return this.isCatching ? COLORS.sliderFill.preciseCatching : COLORS.sliderFill.precise;
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.lineWidth = SIZES.lineWidth.slider;
    ctx.strokeStyle = COLORS.defaultStrokeColor;
    ctx.fillStyle = this.fillColor;
    const width = this.width;

    // Inner Arc
    ctx.beginPath();
    ctx.arc(this.arcCenterX, this.arcCenterY, this.innerRadius,
      this.location + DRAW_ANGLE_OFFSET - width/2,
      this.location + DRAW_ANGLE_OFFSET + width/2);
    // Line joining one end of the two arcs
    const outerArcStartX = (this.arcCenterX +
      this.outerRadius * Math.cos(this.location + DRAW_ANGLE_OFFSET + width/2));
    const outerArcStartY = (this.arcCenterY +
      this.outerRadius * Math.sin(this.location + DRAW_ANGLE_OFFSET + width/2));
    ctx.lineTo(outerArcStartX, outerArcStartY);
    // Outer Arc
    ctx.arc(this.arcCenterX, this.arcCenterY, this.outerRadius,
      this.location + DRAW_ANGLE_OFFSET + width/2, this.location + DRAW_ANGLE_OFFSET - width/2,
      true);
    // Close path to add the line joining the other end of the two arcs
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    // Add line at the middle to indicate slider center location
    ctx.beginPath();
    const innerArcMidX = (this.arcCenterX +
      this.innerRadius * Math.cos(this.location + DRAW_ANGLE_OFFSET));
    const innerArcMidY = (this.arcCenterY +
      this.innerRadius * Math.sin(this.location + DRAW_ANGLE_OFFSET));
    const outerArcMidX = (this.arcCenterX +
      this.outerRadius * Math.cos(this.location + DRAW_ANGLE_OFFSET));
    const outerArcMidY = (this.arcCenterY +
      this.outerRadius * Math.sin(this.location + DRAW_ANGLE_OFFSET));
    ctx.moveTo(innerArcMidX, innerArcMidY);
    ctx.lineTo(outerArcMidX, outerArcMidY);
    ctx.stroke();
    ctx.closePath();
  }
}

class Laser extends MovableDisplay {
  constructor(centerX, centerY, location, hidden = false,
    length = SIZES.radius.laser) {
    super(location);
    this.centerX = centerX;
    this.centerY = centerY;
    this.length = length;
    this.setHiddenWithoutRedraw(hidden);
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    const startX = this.centerX;
    const startY = this.centerY;
    const endX = this.centerX + this.length * Math.cos(this.location + DRAW_ANGLE_OFFSET);
    const endY = this.centerY + this.length * Math.sin(this.location + DRAW_ANGLE_OFFSET);;

    ctx.lineWidth = SIZES.lineWidth.laser;
    ctx.strokeStyle = COLORS.laser;

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.closePath();
  }
}

class TargetLocation extends MovableDisplay {
  constructor(centerX, centerY, location, hidden = false) {
    super(location);
    this.centerX = centerX;
    this.centerY = centerY;
    this.setHiddenWithoutRedraw(hidden);
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    const startX = this.centerX;
    const startY = this.centerY;
    const endX = this.centerX + SIZES.radius.targetLine * Math.cos(this.location + DRAW_ANGLE_OFFSET);
    const endY = this.centerY + SIZES.radius.targetLine * Math.sin(this.location + DRAW_ANGLE_OFFSET);;

    ctx.lineWidth = SIZES.lineWidth.targetLine;
    ctx.strokeStyle = COLORS.target;
    ctx.fillStyle = COLORS.target;
    ctx.setLineDash([2, 1]);

    // line
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.closePath();

    // tip (circle)
    ctx.beginPath();
    ctx.arc(endX, endY, SIZES.radius.targetTip, 0, TWOPI);
    ctx.fill();
    ctx.closePath();

    ctx.setLineDash([]);
  }
}

class ScoreDisplay extends Display {
  constructor(ctx, centerX, scoreBarBottomY, score = INITIAL_VALUES.score) {
    super();
    this._score = score;
    this.scoreBar = new ScoreBar(ctx, centerX, scoreBarBottomY, score);
    this.totalScoreDisplay = new TotalScoreDisplay(ctx, this.scoreBar.barLeftX,
      scoreBarBottomY + LAYOUT.scoreBarBottomLabelBaselineToTotalScoreLabelBaseline,
      score);
  }

  get score() {
    return this._score;
  }

  set score(value) {
    if (value != this._score) {
      this._score = value;
      this.scoreBar.score = value;
      this.totalScoreDisplay.score = value;
    }
  }

  get gainVisible() {
    return this.scoreBar.gainVisible;
  }

  set gainVisible(value) {
    this.scoreBar.gainVisible = value;
  }

  addToScore(gain) {
    this._score += gain;
    this.scoreBar.addToScore(gain);
    this.totalScoreDisplay.score += gain;
  }

  resetScore() {
    this.score = INITIAL_VALUES.score;
    this.scoreBar.gain = 0;
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    this.scoreBar.draw(ctx);
    this.totalScoreDisplay.draw(ctx);
  }
}

class ScoreBar extends Display {
  constructor(ctx, centerX, bottomY,
    score = INITIAL_VALUES.score) {
    super();
    this.centerX = centerX;
    this.bottomY = bottomY;
    this._score = score;
    this._gain = null;
    this._gainVisible = false;
  }

  get score() {
    return this._score;
  }

  set score(value) {
    if (value != this._score) {
      this._score = value;
      setNeedsRedraw();
    }
  }

  get gain() {
    return this._gain;
  }

  set gain(value) {
    if (value != this._gain) {
      this._gain = value;
      setNeedsRedraw();
    }
  }

  addToScore(gain) {
    this.score = this.score + gain;
    this.gain = gain;
  }

  get gainVisible() {
    return this._gainVisible;
  }

  set gainVisible(value) {
    if (value != this._gainVisible) {
      this._gainVisible = value;
      setNeedsRedraw();
    }
  }

  get gainLabelText() {
    const gain = this.gain;
    if (gain === null) {
      return null;
    } else if (gain == 0) {
      return "0";
    } else if (gain > 0) {
      return "+" + gain.toString();
    } else if (gain < 0) {
      return gain.toString();
    }
  }
  
  get scoreBarFrac() {
    if (this.showGainCoin) {
      return 1;
    } else {
      return (this.score % N_POINTS_PER_COIN) / N_POINTS_PER_COIN;
    }
  }

  get gainBarFrac() {
    return this.gain / N_POINTS_PER_COIN;
  }

  get showGainCoin() {
    const prevNCoins = nCoinsWithPoints(this.score - this.gain);
    const nCoins = nCoinsWithPoints(this.score);
    return (this.gainVisible && (prevNCoins < nCoins));
  }

  get showLossCoin() {
    const prevNCoins = nCoinsWithPoints(this.score - this.gain);
    const nCoins = nCoinsWithPoints(this.score);
    return (this.gainVisible && (prevNCoins > nCoins));
  }

  get barLeftX() {
    return this.centerX - Math.ceil(SIZES.width.scoreBar / 2);
  }

  get barRightX() {
    return this.centerX + Math.ceil(SIZES.width.scoreBar / 2);
  }

  get barBottomY() {
    return this.bottomY - LAYOUT.scoreBarBottomToBottomLabelBaseline;
  }

  get barTopY() {
    return this.barBottomY - SIZES.height.scoreBar;
  }

  get barCenterY() {
    return this.barBottomY - Math.round(SIZES.height.scoreBar / 2);
  }

  get topLabelBaselineY() {
    return this.barTopY - LAYOUT.scoreBarTopLabelBaselineToBarTop;
  }


  draw(ctx) {
    if (this.hidden) {
      return;
    }

    // bar
    const fillHeight = Math.round(this.scoreBarFrac * SIZES.height.scoreBar);
    const leftX = this.barLeftX;
    const bottomY = this.barBottomY;
    const gainFillHeight = Math.round(this.gainBarFrac * SIZES.height.scoreBar);
    const gainTopY = bottomY - fillHeight + Math.min(gainFillHeight, 0);
    const gainBottomY = bottomY - fillHeight + Math.max(gainFillHeight, 0);
    ctx.lineWidth = SIZES.lineWidth.scoreBar;
    ctx.fillStyle = COLORS.scoreBar.fill;
    ctx.strokeStyle = COLORS.defaultStrokeColor;
    ctx.fillRect(
      leftX,
      bottomY - fillHeight,
      SIZES.width.scoreBar,
      fillHeight);
    ctx.strokeRect(
      leftX,
      bottomY - fillHeight,
      SIZES.width.scoreBar,
      fillHeight);
    if (this.gainVisible && this.gain != 0) {
      ctx.fillStyle = this.gain > 0 ? COLORS.scoreBar.gainFill : COLORS.scoreBar.lossFill;
      ctx.fillRect(
        leftX,
        gainTopY,
        SIZES.width.scoreBar,
        Math.abs(gainFillHeight));
      ctx.strokeRect(
        leftX,
        gainTopY,
        SIZES.width.scoreBar,
        Math.abs(gainFillHeight)); 
    }
    ctx.strokeRect(
      leftX,
      bottomY - SIZES.height.scoreBar,
      SIZES.width.scoreBar,
      SIZES.height.scoreBar);
    // labels
    drawText(ctx, TEXT.scoreBar.topLabel, this.centerX, this.topLabelBaselineY,
      FONTS.size.scoreBarLabels, "center", "alphabetic",
      COLORS.defaultTextColor, false, FONTS.score);
    drawText(ctx, TEXT.scoreBar.bottomLabel,
      this.centerX, this.bottomY,
      FONTS.size.scoreBarLabels, "center", "alphabetic",
      COLORS.defaultTextColor, false, FONTS.score);
    if (this.gainVisible) {
      const gainLabelText = this.gainLabelText;
      if (gainLabelText) {
        const gainLabelCenterY = (gainTopY + gainBottomY) / 2;
        drawText(ctx, gainLabelText,
          this.barRightX + LAYOUT.scoreBarRightToGainLabelLeft, gainLabelCenterY,
          FONTS.size.gainedPoints, "left", "middle",
          COLORS.defaultTextColor, false, FONTS.score);
      }
    }
    if (this.showGainCoin) {
      drawText(ctx, EMOJI.coin, this.centerX, this.topLabelBaselineY - 14, // TBD
        FONTS.size.gainedCoin,  "center", "alphabetic",
        COLORS.defaultTextColor, false, FONTS.emoji);
    }
    if (this.showLossCoin) {
      drawText(ctx, EMOJI.coin, this.centerX, this.topLabelBaselineY - 14,
        FONTS.size.gainedCoin, "center", "alphabetic",
        COLORS.defaultTextColor, false, FONTS.emoji);
      drawText(ctx, EMOJI.crossmark, this.centerX, this.topLabelBaselineY - 14,
        FONTS.size.gainedCoin, "center", "alphabetic",
        COLORS.defaultTextColor, false, FONTS.emoji);
    }
  }
}

class TotalScoreDisplay extends Display {
  constructor(ctx, leftX, topAnchorY,
    score = INITIAL_VALUES.score) {
    super();
    this.leftX = leftX;
    this.topAnchorY = topAnchorY;
    this._score = score;
  }

  get score() {
    return this._score;
  }

  set score(value) {
    if (value != this._score) {
      this._score = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    drawText(ctx,
      TEXT.totalScoreLabel(this.score),
      this.leftX, this.topAnchorY,
      FONTS.size.totalScore,  "left", "alphabetic",
      COLORS.defaultTextColor, false, FONTS.score);

    // coins
    const nCoins = nCoinsWithPoints(this.score);
    if (nCoins > 0) {
      const coinsLabelTopY = this.topAnchorY + LAYOUT.totalScoreLabelBaselineToCoinsLabelTop;
      const coinsMultilineLabel = new MultilineTextDisplay(ctx, this.leftX, coinsLabelTopY,
        TEXT.coinsLabel(nCoins), SIZES.width.totalScoreDisplay, "left", "top", FONTS.emoji,
        FONTS.size.totalCoins, COLORS.defaultTextColor, 1.4);
      coinsMultilineLabel.draw(ctx);
    }
  }
}

export function getCanvas() {
  const canvas = document.getElementById("canvas");
  canvas.widthCSS = canvas.width;
  canvas.heightCSS = canvas.height;
  canvas.adaptToScreenDPI = function() {
    // The following code makes the canvas as sharp as possible for the device,
    // taking into account the screen DPI such that the canvas is drawn
    // at a higher resolution than its CSS size when the device pixel ratio
    // is greater than 1 (i.e., one CSS pixel corresponds to multiple device pixels),
    // such as on Retina screens.
    // See
    // <https://www.kirupa.com/canvas/canvas_high_dpi_retina.html>
    // <https://gist.github.com/callumlocke/cc258a193839691f60dd>
    const canvasWidthCss = this.widthCSS;
    const canvasHeightCss = this.heightCSS;
    // let canvasWidthCss = canvas.width;
    // let canvasHeightCss = canvas.height;
    let pxRatioPhysToCss = window.devicePixelRatio;
    this.width = canvasWidthCss * pxRatioPhysToCss;
    this.height = canvasHeightCss * pxRatioPhysToCss;
    this.style.width = canvasWidthCss + 'px';
    this.style.height = canvasHeightCss + 'px';
    this.getContext("2d").scale(pxRatioPhysToCss, pxRatioPhysToCss);
  }
  Object.defineProperty(canvas, "hidden", getHiddenPropertyDescriptor());
  return canvas;
}

export function getTextBox() {
  const rootElem = document.getElementById("text-box");
  const blockCompletedElem = rootElem.querySelector("#block-completed");
  const totalBlocksElem = rootElem.querySelector("#total-blocks");
  const pointsElem = rootElem.querySelector("#points");
  const poundsElem = rootElem.querySelector("#pounds");
  const nextBlockTextElem = rootElem.querySelector("#next-block-text");
  const finalBlockTextElem = rootElem.querySelector("#final-block-text");
  const totalPointsElem = rootElem.querySelector("#total-points");
  const totalPoundsElem = rootElem.querySelector("#total-pounds");
  const prolificCompletionLinkElem = rootElem.querySelector("#prolific-completion-link");
  const savingDataElem = rootElem.querySelector("#saving-data");
  for (const elem of [rootElem, nextBlockTextElem, finalBlockTextElem,
    prolificCompletionLinkElem, savingDataElem]) {
    Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  }
  rootElem.setBlocksCompleted = function(val) {
    blockCompletedElem.textContent = val.toString();
  }
  rootElem.setTotalBlocks = function(val) {
    totalBlocksElem.textContent = val.toString();
  }
  rootElem.setPoints = function(val) {
    pointsElem.textContent = val.toString();
  }
  rootElem.setPounds = function(val) {
    poundsElem.textContent = val.toFixed(2).toString();
  }
  rootElem.setTotalPoints = function(val) {
    totalPointsElem.textContent = val.toString();
  }
  rootElem.setTotalPounds = function(val) {
    totalPoundsElem.textContent = val.toFixed(2).toString();
  }
  rootElem.showNextBlockText = function() {
    finalBlockTextElem.hidden = true;
    nextBlockTextElem.hidden = false;
  }
  rootElem.showFinalBlockText = function() {
    nextBlockTextElem.hidden = true;
    finalBlockTextElem.hidden = false;
  }
  rootElem.showProlificCompletionLink = function(url) {
    prolificCompletionLinkElem.hidden = false;
    prolificCompletionLinkElem.href = url
  }
  rootElem.showSavingData = function() {
    savingDataElem.hidden = false;
  }
  rootElem.hideSavingData = function() {
    savingDataElem.hidden = true;
  }

  return rootElem;
}

export function getInstructionContainer() {
  const elem = document.getElementById("instruction-container");
  Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  return elem;
}

export function getConsentContainer() {
  const elem = document.getElementById("consent-container");
  Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  return elem;
}

export function getHelpPanel() {
  const elem = document.getElementById("help-panel");
  Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  return elem;
}

export function getPracticeHelpPanel() {
  const elem = document.getElementById("practice-help-panel");
  Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  return elem;
}

export function getFeedbackForm() {
  const elem = document.getElementById("feedback-form");
  Object.defineProperty(elem, "hidden", getHiddenPropertyDescriptor());
  return elem;
}

function getHiddenPropertyDescriptor() {
  return {
    get() {
      return this.style.display == "none";
    },
    set(value) {
      if (value) {
        this.style.display = "none";
      } else {
        this.style.display = "";
      }
    },
  };
}