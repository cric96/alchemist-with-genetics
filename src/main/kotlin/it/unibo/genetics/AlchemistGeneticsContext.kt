package it.unibo.genetics

import it.unibo.alchemist.loader.Loader
import it.unibo.alchemist.model.interfaces.Position

class AlchemistGeneticsContext<T, P : Position<P>, G>(
    val loader: Loader,
    val injectMolecule: GenotypeInjection<G>,
    val fitnessEvaluation: FitnessEvaluation<T, P>,
    val endCondition: EndCondition<G>
)