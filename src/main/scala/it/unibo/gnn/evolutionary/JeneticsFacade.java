package it.unibo.gnn.evolutionary;

import io.jenetics.Chromosome;
import io.jenetics.Gene;
import io.jenetics.Genotype;

import java.util.Arrays;

public class JeneticsFacade {
    static <G extends Gene<?, G>> Genotype<G> of(Chromosome<G> ... chromosomes) {
        return Genotype.of(Arrays.asList(chromosomes));
    }
}
