package it.unibo.gnn.app

import it.unibo.scafi.simulation.frontend.SettingsSpace.Topologies
import it.unibo.scafi.simulation.frontend.{Launcher, Settings}


trait VisualizationTest extends Launcher {
  Settings.ShowConfigPanel = false
  Settings.Sim_NbrRadius = 0.4
  Settings.Sim_NumNodes = 9
  Settings.Sim_Topology = Topologies.Grid
}

object NonLinearTest extends VisualizationTest {
  Settings.Sim_ProgramClass = "it.unibo.gnn.app.program.NonLinearGNNVisual"
  launch()
}

object LinearTest extends VisualizationTest {
  Settings.Sim_ProgramClass = "it.unibo.gnn.app.program.LinearGNNVisual"
  launch()
}
