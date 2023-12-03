"use strict";

let context = window.ContextModel;

async function init(
  optimizedStructureUrl,
  originalStructureUrl,
  residueLogsUrl
) {
  context.init(optimizedStructureUrl, originalStructureUrl, residueLogsUrl);

  mountViewControls();
  mountColorControls();

  const bas = document.getElementById("view_bas");
  const structure = document.getElementById("colors_structure");
  const nonOptimized = document.getElementById("non_optimized");
  const view = document.getElementById("view_fieldset");
  if (!bas || !structure || !nonOptimized || !view) {
    console.error("Controls not found");
    return;
  }
  bas.setAttribute("checked", "true");
  structure.setAttribute("checked", "true");
  nonOptimized.setAttribute("checked", "true");
  view.addEventListener("change", (e) => {
    if (!bas.checked) {
      nonOptimized.setAttribute("disabled", "true");
    } else {
      nonOptimized.removeAttribute("disabled");
    }
  });
}

function mountViewControls() {
  const cartoon = document.getElementById("view_cartoon");
  const surface = document.getElementById("view_surface");
  const bas = document.getElementById("view_bas");
  const nonOptimized = document.getElementById("non_optimized");
  if (!cartoon || !surface || !bas || !nonOptimized) {
    console.error("View controls not found");
    return;
  }

  cartoon.onclick = async () => await context.changeView("cartoon");
  surface.onclick = async () => await context.changeView("gaussian-surface");
  bas.onclick = async () => await context.changeView("ball-and-stick");
  nonOptimized.onclick = async () => await context.toggleVisibility();
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
}
