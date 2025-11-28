// 
// Mouse event handling
// 

export class MouseTracker {
  constructor(updateForMouseMovement,
    getCurrentMouseTargets,
    mouseTargetEnabled = true) {
    this.isTrackingMouseMovement = false;
    this._trackingUpdateEnabled = true;
    this._mouseTargetEnabled = mouseTargetEnabled;
    this.updateForMouseMovement = updateForMouseMovement;
    this.getCurrentMouseTargets = getCurrentMouseTargets;
  }

  get domElement() {
    return this._domElement;
  }

  set domElement(domElement) {
    this._domElement = domElement;
    domElement.onmousemove = (e) => { this.mouseDidMove(e) };
    domElement.onmousedown = (e) => { this.mouseDidPressDown(e) };
    domElement.onmouseup = (e) => { this.mouseDidPressUp(e) };
    domElement.addEventListener("dblclick", (e) => { this.mouseDidDoubleClick(e) }, true);
  }

  mouseDidMove(mouseEvent) {
    if (this.isTrackingMouseMovement && this._trackingUpdateEnabled) {
      this.updateForMouseMovement?.(mouseEvent);
    }
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      let mouseIsInsideAnyTarget = false;
      for (let target of mouseTargets) {
        const mouseWasInsideTarget = target.mouseIsInside;
        const mouseIsInsideTarget = target.isInside(mouseX, mouseY);
        if (mouseIsInsideTarget && !mouseWasInsideTarget) {
          target.mouseDidEnter?.();
        }
        if (mouseIsInsideTarget && mouseWasInsideTarget) {
          target.mouseDidMove?.();
        }
        if (!mouseIsInsideTarget && mouseWasInsideTarget) {
          target.mouseDidExit?.();
        }
        target.mouseIsInside = mouseIsInsideTarget;
        mouseIsInsideAnyTarget = (mouseIsInsideAnyTarget || mouseIsInsideTarget);
      }
      // if (mouseIsInsideAnyTarget) {
        // this.domElement.style.cursor = "pointer";
      // } else {
        // this.domElement.style.cursor = "auto";
      // }
    }
  }
  
  get areMouseTargetsEnabled() {
    // return !this.isTrackingMouseMovement;
    return this._mouseTargetEnabled;
  }

  enableMouseTargets(enabled) {
    this._mouseTargetEnabled = enabled;
  }

  mouseDidPressDown(mouseEvent) {
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      for (let target of mouseTargets) {
        if (target.isInside(mouseX, mouseY)) {
          target?.mouseDidPressDown(mouseEvent);
        }
      }
    }
  }

  mouseDidPressUp(mouseEvent) {
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      for (let target of mouseTargets) {
        if (target.isInside(mouseX, mouseY)) {
          target?.mouseDidPressUp(mouseEvent);
        }
      }
    }
  }

  mouseDidDoubleClick(mouseEvent) {
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      for (let target of mouseTargets) {
        if (target.isInside(mouseX, mouseY)) {
          target?.mouseDidDoubleClick(mouseEvent);
        }
      }
    }
  }

  startTrackingMouseMovement() {
    this.isTrackingMouseMovement = true;
    // this.domElement.style.cursor = "none";
    this.domElement.style.cursor = "grabbing";
    // this.domElement.style.cursor = "none";
  }

  stopTrackingMouseMovement() {
    this.isTrackingMouseMovement = false;
    // this.domElement.style.cursor = "auto";
    this.domElement.style.cursor = "grab";
  }

  disableTrackingUpdates() {
    this._trackingUpdateEnabled = false;
  }

  enableTrackingUpdates() {
    this._trackingUpdateEnabled = true;
  }
}

export class MouseTarget {
  constructor(
    // canvas,
    clickAction,
    rightClickAction = null,
    metaClickAction = null,
    altClickAction = null,
    doubleClickAction = null,
    enabled = true) {
    this.clickAction = clickAction;
    this.rightClickAction = rightClickAction;
    this.metaClickAction = metaClickAction;
    this.altClickAction = altClickAction;
    this.doubleClickAction = doubleClickAction;
    this.enabled = enabled;
    this.pressed = false;
    this.rightButtonPressed = false;
    this._highlighted = false;
  }

  isInside(x, y) {
    if (this.enabled) {
      return true;
    } else {
      return false;
    }
  }

  mouseDidPressDown(mouseEvent) {
    if (mouseEvent.button == 2) {
      this.rightButtonPressed = true;
    } else {
      this.pressed = true;
    }
  }

  mouseDidPressUp(mouseEvent) {
    if (this.pressed) {
      this.mouseDidClick(mouseEvent.metaKey, mouseEvent.altKey);
    } else if (this.rightButtonPressed && (mouseEvent.button == 2)) {
      this.mouseDidRightClick();
    }
    this.pressed = false;
  }

  mouseDidClick(metaKey = false, altKey = false) {
    this.highlighted = false;
    if (metaKey && this.metaClickAction) {
      this.metaClickAction();
    } else if (altKey && this.altClickAction) {
      this.altClickAction();
    } else {
      this.clickAction?.();
    }
  }

  mouseDidRightClick() {
    this.rightClickAction?.();
  }

  mouseDidDoubleClick() {
    this.doubleClickAction?.();
  }

  get highlighted() {
    return this._highlighted;
  }

  set highlighted(value) {
    if (value != this._highlighted) {
      this._highlighted = value;
    }
  }
}