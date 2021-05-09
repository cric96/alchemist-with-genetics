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
class AlchemistScoreCalculation(private val file : String, val maxTime : Double = 1.0) : CalculateScore {
    private val regressionMolecule = "regression"
    override fun calculateScore(method: MLMethod?): Double {
        val alchemist = LoadAlchemist.from(ClassLoader.getSystemClassLoader().getResource(file)!!)
        val simulation = Engine(alchemist.getDefault<Any, Position<*>>().environment, DoubleTime(maxTime))
        when(method) {
            is MLRegression -> {
                simulation.environment.nodes.forEach { it.setConcentration(SimpleMolecule(regressionMolecule), method) }
                simulation.play()
                simulation.run()
                val error = simulation.environment.nodes
                        .map { node -> Math.abs(node.getDouble("input0") * node.getDouble("input1") - node.getDouble("output")) }
                        .sum()
                return error
            }
            else -> throw IllegalArgumentException("Alchemist accept only ml regression")
        }
    }

    override fun shouldMinimize(): Boolean { return true }

    //for now..
    override fun requireSingleThreaded(): Boolean { return true }

    fun Node<Any>.getDouble(name : String) : Double {
        when(val molecule = this.getConcentration(SimpleMolecule(name))) {
            is Double -> return molecule
            else -> throw java.lang.IllegalArgumentException("molecule isn't a double")
        }
    }
}