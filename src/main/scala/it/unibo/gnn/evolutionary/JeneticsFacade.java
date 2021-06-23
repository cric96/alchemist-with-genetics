package it.unibo.gnn.evolutionary;

import io.jenetics.Chromosome;
import io.jenetics.Gene;
import io.jenetics.Genotype;
import io.jenetics.engine.Engine;
import io.jenetics.util.Factory;

import java.util.Arrays;
import java.util.function.Function;

public class JeneticsFacade {
    public static <G extends Gene<?, G>> Genotype<G> of(Chromosome<G> ... chromosomes) {
        return Genotype.of(Arrays.asList(chromosomes));
    }
    public static <G extends Gene<?, G>> Engine.Builder<G, Double> doubleEngine(Function<Genotype<G>, Double> fitness, Factory<Genotype<G>> factory) {
        return Engine.builder(fitness, factory);
    }

}
