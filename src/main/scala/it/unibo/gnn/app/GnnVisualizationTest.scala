package it.unibo.gnn.app

import it.unibo.scafi.simulation.frontend.SettingsSpace.Topologies
import it.unibo.scafi.simulation.frontend.{Launcher, Settings}

object GnnVisualizationTest extends Launcher {
  Settings.Sim_ProgramClass = "it.unibo.gnn.app.programs.ScafiHopCountGNNVisual"
  Settings.ShowConfigPanel = false
  Settings.Sim_NbrRadius = 0.4
  Settings.Sim_NumNodes = 9
  Settings.Sim_Topology = Topologies.Grid
  launch()
}
