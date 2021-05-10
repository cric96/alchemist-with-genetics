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

class AndScoreCalculation(file : String, maxTime : Double = 1.0) : AbstractAlchemistScoreCalculation(file, maxTime) {
    override fun evalSimulation(simulation: Engine<Any, Position<*>>): Double {
        return simulation.environment.nodes.sumOf { node ->
            abs(node.getDouble("input0") * node.getDouble("input1") - node.getDouble("output"))
        }
    }

    fun Node<Any>.getDouble(name : String) : Double {
        when(val molecule = this.getConcentration(SimpleMolecule(name))) {
            is Double -> return molecule
            else -> throw java.lang.IllegalArgumentException("molecule isn't a double")
        }
    }
}