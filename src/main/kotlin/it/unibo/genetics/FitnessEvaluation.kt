package it.unibo.genetics

import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.Position

interface FitnessEvaluation<T, P : Position<P>> {
    fun score(environment : Environment<T, P>) : Double
}