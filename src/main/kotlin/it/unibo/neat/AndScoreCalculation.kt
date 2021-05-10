package it.unibo.neat

import it.unibo.alchemist.core.implementations.Engine
import it.unibo.alchemist.loader.LoadAlchemist
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.interfaces.Position
import org.encog.ml.CalculateScore
import org.encog.ml.MLMethod
import org.encog.ml.MLRegression
import kotlin.math.abs

class AndScoreCalculation(file : String, maxTime : Double = 1.0) : AbstractAlchemistScoreCalculation(file, maxTime), PimpNode {
    override fun evalSimulation(simulation: Engine<Any, Position<*>>): Double {
        return simulation.environment.nodes.sumOf { node ->
            abs(node.dataDouble("input0") * node.dataDouble("input1") - node.dataDouble("output"))
        }
    }
}