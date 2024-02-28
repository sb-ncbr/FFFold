"use strict";

let context = window.ContextModel;

async function init(
  optimisedStructureUrl,
  originalStructureUrl,
  residueLogsUrl
) {
  await context.init(optimisedStructureUrl, originalStructureUrl, residueLogsUrl);

  mountViewControls();
  mountColorControls();
  disableMolstarButtons();

  const bas = document.getElementById("view_bas");
  const structure = document.getElementById("colors_structure");
  const nonOptimised = document.getElementById("non_optimised");
  const view = document.getElementById("view_fieldset");
  if (!bas || !structure || !nonOptimised || !view) {
    console.error("Controls not found");
    return;
  }
  bas.setAttribute("checked", "true");
  structure.setAttribute("checked", "true");
  nonOptimised.setAttribute("checked", "true");
  view.addEventListener("change", (e) => {
    if (!bas.checked) {
      nonOptimised.setAttribute("disabled", "true");
    } else {
      nonOptimised.removeAttribute("disabled");
    }
  });
}

function mountViewControls() {
  const cartoon = document.getElementById("view_cartoon");
  const surface = document.getElementById("view_surface");
  const bas = document.getElementById("view_bas");
  const nonOptimised = document.getElementById("non_optimised");
  if (!cartoon || !surface || !bas || !nonOptimised) {
    console.error("View controls not found");
    return;
  }

  cartoon.onclick = async () => await context.changeView("cartoon");
  surface.onclick = async () => await context.changeView("gaussian-surface");
  bas.onclick = async () => await context.changeView("ball-and-stick");
  nonOptimised.onclick = async () => await context.toggleVisibility();

  cartoon.removeAttribute("disabled");
  surface.removeAttribute("disabled");
  bas.removeAttribute("disabled");
  nonOptimised.removeAttribute("disabled");
}

function mountColorControls() {
  const structure = document.getElementById("colors_structure");
  const alphafold = document.getElementById("colors_alphafold");
  if (!structure || !alphafold) {
    console.error("Color controls not found");
    return;
  }
  structure.onclick = async () => await context.changeColor("element-symbol");
  alphafold.onclick = async () => await context.changeColor("plddt-confidence");

  structure.removeAttribute("disabled");
  alphafold.removeAttribute("disabled");
}

function disableMolstarButtons() {
  const settingsButton = document.querySelector(
    "#root > div > div.relative.grow > div > div.msp-viewport-controls > div > div:nth-child(3) > button.msp-btn.msp-btn-icon.msp-btn-link-toggle-on"
  );
  const expandButton = document.querySelector(
    "#root > div > div.relative.grow > div > div.msp-viewport-controls > div > div:nth-child(3) > button:nth-child(3)"
  );

  if (!settingsButton || !expandButton) {
    console.error("Molstar buttons not found");
    return;
  }

  settingsButton.setAttribute("disabled", "true");
  expandButton.setAttribute("disabled", "true");
}
