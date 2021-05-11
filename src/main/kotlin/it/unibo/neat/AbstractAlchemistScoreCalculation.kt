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

abstract class AbstractAlchemistScoreCalculation(private val file : String, val maxTime : Double = 1.0, private val regressionMolecule: String = "regression") : CalculateScore {
    private val alchemist = LoadAlchemist.from(ClassLoader.getSystemClassLoader().getResource(file)!!)
    override fun calculateScore(method: MLMethod?): Double {
        val simulation = synchronized(this) {
            Engine(alchemist.getDefault<Any, Position<*>>().environment, DoubleTime(maxTime))
        }
        when(method) {
            is MLRegression -> {
                simulation.environment.nodes.forEach { it.setConcentration(SimpleMolecule(regressionMolecule), method) }
                simulation.play()
                simulation.run()
                return evalSimulation(simulation)
            }
            else -> throw IllegalArgumentException("Alchemist accept only ml regression")
        }
    }

    override fun shouldMinimize(): Boolean { return true }

    override fun requireSingleThreaded(): Boolean { return false }

    protected abstract fun evalSimulation(simulation : Engine<Any, Position<*>>) : Double
}