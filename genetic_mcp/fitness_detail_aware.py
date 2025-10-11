"""Detail-aware fitness metrics for evaluating implementation depth and actionability.

This module implements mathematical metrics for assessing the level of detail in generated ideas:
- Implementation depth (δ): Measures concrete implementation details
- Actionability (α): Measures the presence of actionable steps
- Completeness (κ): Measures coverage of necessary components
- Technical precision (τ): Measures technical terminology and specificity

Mathematical Formulations:
========================

1. Implementation Depth (δ):
   δ(idea) = (w_t * τ_tech + w_s * σ_struct + w_e * ε_examples) / (w_t + w_s + w_e)

   where:
   - τ_tech: Technical term density = count(technical_terms) / total_words
   - σ_struct: Structural depth = depth_score(code_blocks, diagrams, equations)
   - ε_examples: Example density = count(concrete_examples) / total_sections
   - w_t, w_s, w_e: weights (default: 0.4, 0.3, 0.3)

2. Actionability (α):
   α(idea) = (1/n) * Σ(score(step_i) * weight(step_i))

   where:
   - score(step_i) = 1 if step is actionable, 0 otherwise
   - weight(step_i) = specificity of step (0 to 1)
   - Actionable step indicators: verbs like "implement", "create", "define", etc.

3. Completeness (κ):
   κ(idea) = |components_present| / |components_required|

   where:
   - components_present: detected components (problem, solution, implementation, testing, etc.)
   - components_required: expected components based on idea type
   - Coverage bonus for balanced distribution

4. Technical Precision (τ):
   τ(idea) = (w_d * density + w_v * variety + w_c * context) / (w_d + w_v + w_c)

   where:
   - density: Technical terms per sentence
   - variety: Unique technical terms / total technical terms
   - context: Proportion of terms used in technically appropriate context
   - w_d, w_v, w_c: weights (default: 0.4, 0.3, 0.3)
"""

import re

import numpy as np

from .logging_config import setup_logging
from .models import Idea

logger = setup_logging(component="detail_metrics")


class DetailMetrics:
    """Calculate detail-aware metrics for ideas."""

    # Technical terms for various domains
    TECHNICAL_TERMS = {
        'programming': {
            'algorithm', 'function', 'class', 'method', 'variable', 'api', 'database',
            'framework', 'library', 'module', 'interface', 'implementation', 'parameter',
            'return', 'exception', 'inheritance', 'polymorphism', 'encapsulation',
            'authentication', 'authorization', 'middleware', 'endpoint', 'query',
            'schema', 'migration', 'deployment', 'pipeline', 'repository', 'commit',
            'branch', 'merge', 'refactor', 'optimize', 'debug', 'test', 'mock',
            'stub', 'fixture', 'assertion', 'coverage', 'integration', 'unit',
            'async', 'concurrent', 'parallel', 'thread', 'process', 'mutex', 'lock'
        },
        'data_science': {
            'dataset', 'feature', 'model', 'training', 'validation', 'testing',
            'accuracy', 'precision', 'recall', 'f1-score', 'confusion matrix',
            'cross-validation', 'hyperparameter', 'gradient', 'loss function',
            'optimization', 'regularization', 'overfitting', 'underfitting',
            'bias', 'variance', 'ensemble', 'bagging', 'boosting', 'neural network',
            'deep learning', 'regression', 'classification', 'clustering',
            'dimensionality reduction', 'pca', 'embedding', 'tokenization',
            'normalization', 'standardization', 'pipeline', 'preprocessing'
        },
        'mathematics': {
            'equation', 'theorem', 'proof', 'lemma', 'corollary', 'axiom',
            'function', 'derivative', 'integral', 'limit', 'convergence',
            'divergence', 'matrix', 'vector', 'eigenvalue', 'eigenvector',
            'determinant', 'rank', 'span', 'basis', 'dimension', 'kernel',
            'image', 'manifold', 'topology', 'metric', 'norm', 'inner product',
            'probability', 'distribution', 'expectation', 'variance', 'covariance',
            'independence', 'conditional', 'bayesian', 'likelihood', 'posterior'
        },
        'general': {
            'system', 'component', 'architecture', 'design', 'pattern',
            'principle', 'requirement', 'specification', 'constraint',
            'objective', 'metric', 'performance', 'scalability', 'reliability',
            'maintainability', 'security', 'efficiency', 'complexity',
            'tradeoff', 'optimization', 'analysis', 'evaluation'
        }
    }

    # Actionable verbs indicating concrete steps
    ACTIONABLE_VERBS = {
        'implement', 'create', 'define', 'build', 'develop', 'design',
        'configure', 'setup', 'install', 'initialize', 'instantiate',
        'execute', 'run', 'test', 'validate', 'verify', 'measure',
        'optimize', 'refactor', 'modify', 'update', 'add', 'remove',
        'deploy', 'integrate', 'connect', 'establish', 'construct',
        'generate', 'produce', 'calculate', 'compute', 'process',
        'transform', 'convert', 'parse', 'serialize', 'deserialize',
        'encode', 'decode', 'encrypt', 'decrypt', 'compress', 'decompress'
    }

    # Component types for completeness assessment
    COMPONENT_TYPES = {
        'problem_definition': [
            r'\bproblem\b', r'\bissue\b', r'\bchallenge\b', r'\bgoal\b',
            r'\bobjective\b', r'\brequirement\b'
        ],
        'solution_approach': [
            r'\bsolution\b', r'\bapproach\b', r'\bmethod\b', r'\bstrategy\b',
            r'\btechnique\b', r'\balgorithm\b'
        ],
        'implementation_details': [
            r'\bimplement\b', r'\bcode\b', r'\bfunction\b', r'\bclass\b',
            r'\bmodule\b', r'\bapi\b', r'\bstep\s+\d+', r'\b\d+\.'
        ],
        'technical_specification': [
            r'\bdata\s+type\b', r'\bparameter\b', r'\breturn\s+type\b',
            r'\binput\b', r'\boutput\b', r'\bformat\b', r'\bschema\b'
        ],
        'testing_validation': [
            r'\btest\b', r'\bvalidat\b', r'\bverif\b', r'\bassert\b',
            r'\bcheck\b', r'\bensure\b', r'\bexpect\b'
        ],
        'examples_demonstrations': [
            r'\bexample\b', r'\binstance\b', r'\bdemonstrat\b', r'\bsample\b',
            r'\bfor\s+example\b', r'\be\.g\.\b', r'\bsuch\s+as\b'
        ],
        'considerations_tradeoffs': [
            r'\btradeoff\b', r'\blimitation\b', r'\bconsider\b', r'\bcaveat\b',
            r'\bpros?\s+and\s+cons?\b', r'\badvantage\b', r'\bdisadvantage\b'
        ]
    }

    def __init__(self):
        """Initialize detail metrics calculator."""
        # Compile regex patterns for efficiency
        self._code_block_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self._numbered_list_pattern = re.compile(r'^\s*\d+\.', re.MULTILINE)
        self._bullet_list_pattern = re.compile(r'^\s*[-*•]', re.MULTILINE)
        self._technical_term_cache: dict[str, set[str]] = {}

        # Compile component patterns
        self._component_patterns = {
            comp: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for comp, patterns in self.COMPONENT_TYPES.items()
        }

    def calculate_implementation_depth(self, idea: Idea) -> float:
        """Calculate implementation depth score δ(idea).

        δ(idea) = (w_t * τ_tech + w_s * σ_struct + w_e * ε_examples) / (w_t + w_s + w_e)

        Args:
            idea: The idea to evaluate

        Returns:
            Implementation depth score in [0, 1]
        """
        content = idea.content.lower()
        words = content.split()
        total_words = len(words)

        if total_words == 0:
            return 0.0

        # Calculate τ_tech: Technical term density
        technical_terms = self._extract_technical_terms(content)
        tau_tech = min(len(technical_terms) / max(total_words, 1), 1.0)

        # Calculate σ_struct: Structural depth
        sigma_struct = self._calculate_structural_depth(idea.content)

        # Calculate ε_examples: Example density
        epsilon_examples = self._calculate_example_density(idea.content)

        # Weighted combination
        w_t, w_s, w_e = 0.4, 0.3, 0.3
        delta = (w_t * tau_tech + w_s * sigma_struct + w_e * epsilon_examples) / (w_t + w_s + w_e)

        logger.debug(f"Implementation depth for idea {idea.id}: δ={delta:.3f} "
                    f"(τ_tech={tau_tech:.3f}, σ_struct={sigma_struct:.3f}, ε_examples={epsilon_examples:.3f})")

        return float(np.clip(delta, 0.0, 1.0))

    def calculate_actionability(self, idea: Idea) -> float:
        """Calculate actionability score α(idea).

        α(idea) = (1/n) * Σ(score(step_i) * weight(step_i))

        Args:
            idea: The idea to evaluate

        Returns:
            Actionability score in [0, 1]
        """
        content = idea.content.lower()

        # Extract sentences and potential steps
        sentences = re.split(r'[.!?]+', content)

        # Find numbered steps and bullet points
        numbered_steps = self._numbered_list_pattern.findall(idea.content)
        bullet_steps = self._bullet_list_pattern.findall(idea.content)

        total_steps = len(numbered_steps) + len(bullet_steps)

        # Score each sentence/step for actionability
        actionable_count = 0
        weighted_scores = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Check for actionable verbs
            has_actionable_verb = any(verb in sentence for verb in self.ACTIONABLE_VERBS)

            if has_actionable_verb:
                # Calculate specificity weight
                specificity = self._calculate_specificity(sentence)
                weighted_scores.append(specificity)
                actionable_count += 1

        # Calculate alpha
        if not sentences:
            return 0.0

        alpha = np.mean(weighted_scores) if weighted_scores else 0.0

        # Bonus for having structured steps
        structure_bonus = 0.0
        if total_steps > 0:
            structure_bonus = min(0.2, total_steps * 0.05)
            alpha = min(1.0, alpha + structure_bonus)

        logger.debug(f"Actionability for idea {idea.id}: α={alpha:.3f} "
                    f"(actionable_steps={actionable_count}/{len(sentences)}, structure_bonus={structure_bonus:.3f})")

        return float(np.clip(alpha, 0.0, 1.0))

    def calculate_completeness(self, idea: Idea) -> float:
        """Calculate completeness score κ(idea).

        κ(idea) = |components_present| / |components_required|

        Args:
            idea: The idea to evaluate

        Returns:
            Completeness score in [0, 1]
        """
        content = idea.content

        # Detect present components
        components_present = set()
        component_counts = {}

        for component, patterns in self._component_patterns.items():
            count = 0
            for pattern in patterns:
                matches = pattern.findall(content)
                count += len(matches)

            if count > 0:
                components_present.add(component)
                component_counts[component] = count

        # Calculate base completeness
        total_components = len(self.COMPONENT_TYPES)
        present_count = len(components_present)

        kappa = present_count / total_components

        # Bonus for balanced distribution
        if component_counts:
            counts = list(component_counts.values())
            balance_score = 1.0 - (np.std(counts) / (np.mean(counts) + 1e-6))
            balance_bonus = min(0.15, balance_score * 0.15)
            kappa = min(1.0, kappa + balance_bonus)

        # Store component analysis in metadata
        idea.metadata['completeness_components'] = {
            'present': list(components_present),
            'counts': component_counts,
            'coverage': present_count / total_components
        }

        logger.debug(f"Completeness for idea {idea.id}: κ={kappa:.3f} "
                    f"(components={present_count}/{total_components})")

        return float(np.clip(kappa, 0.0, 1.0))

    def calculate_technical_precision(self, idea: Idea) -> float:
        """Calculate technical precision score τ(idea).

        τ(idea) = (w_d * density + w_v * variety + w_c * context) / (w_d + w_v + w_c)

        Args:
            idea: The idea to evaluate

        Returns:
            Technical precision score in [0, 1]
        """
        content = idea.content.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]

        if not sentences:
            return 0.0

        # Extract technical terms
        technical_terms = self._extract_technical_terms(content)
        technical_terms_list = list(technical_terms)

        if not technical_terms_list:
            return 0.0

        # Calculate density: Technical terms per sentence
        density = len(technical_terms_list) / len(sentences)
        density_normalized = min(density / 3.0, 1.0)  # Normalize assuming ~3 terms/sentence is high

        # Calculate variety: Unique terms / total terms
        total_term_occurrences = sum(content.count(term) for term in technical_terms)
        variety = len(technical_terms) / max(total_term_occurrences, 1)

        # Calculate context: Proportion of terms used in appropriate context
        # Check if technical terms appear with related terms (simple heuristic)
        context_score = self._calculate_context_score(content, technical_terms_list)

        # Weighted combination
        w_d, w_v, w_c = 0.4, 0.3, 0.3
        tau = (w_d * density_normalized + w_v * variety + w_c * context_score) / (w_d + w_v + w_c)

        logger.debug(f"Technical precision for idea {idea.id}: τ={tau:.3f} "
                    f"(density={density_normalized:.3f}, variety={variety:.3f}, context={context_score:.3f})")

        return float(np.clip(tau, 0.0, 1.0))

    def calculate_detail_score(self, idea: Idea, weights: dict[str, float] | None = None) -> float:
        """Calculate combined detail score from all metrics.

        Args:
            idea: The idea to evaluate
            weights: Optional weights for combining metrics (delta, alpha, kappa, tau)
                    Default: equal weights (0.25 each)

        Returns:
            Combined detail score in [0, 1]
        """
        if weights is None:
            weights = {'delta': 0.25, 'alpha': 0.25, 'kappa': 0.25, 'tau': 0.25}

        # Calculate individual metrics
        delta = self.calculate_implementation_depth(idea)
        alpha = self.calculate_actionability(idea)
        kappa = self.calculate_completeness(idea)
        tau = self.calculate_technical_precision(idea)

        # Store individual metrics in metadata
        idea.metadata['detail_metrics'] = {
            'implementation_depth': delta,
            'actionability': alpha,
            'completeness': kappa,
            'technical_precision': tau
        }

        # Combine with weights
        detail_score = (
            weights.get('delta', 0.25) * delta +
            weights.get('alpha', 0.25) * alpha +
            weights.get('kappa', 0.25) * kappa +
            weights.get('tau', 0.25) * tau
        )

        logger.debug(f"Detail score for idea {idea.id}: {detail_score:.3f} "
                    f"(δ={delta:.3f}, α={alpha:.3f}, κ={kappa:.3f}, τ={tau:.3f})")

        return float(np.clip(detail_score, 0.0, 1.0))

    # Helper methods

    def _extract_technical_terms(self, content: str) -> set[str]:
        """Extract technical terms from content."""
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))

        # Find all technical terms across domains
        all_technical_terms = set()
        for domain_terms in self.TECHNICAL_TERMS.values():
            all_technical_terms.update(domain_terms)

        # Find matches
        found_terms = words & all_technical_terms

        # Also check for multi-word technical terms
        for domain_terms in self.TECHNICAL_TERMS.values():
            for term in domain_terms:
                if ' ' in term and term in content_lower:
                    found_terms.add(term)

        return found_terms

    def _calculate_structural_depth(self, content: str) -> float:
        """Calculate structural depth score."""
        score = 0.0

        # Code blocks (highest weight)
        code_blocks = self._code_block_pattern.findall(content)
        if code_blocks:
            score += min(len(code_blocks) * 0.3, 0.6)

        # Numbered lists
        numbered_items = self._numbered_list_pattern.findall(content)
        if numbered_items:
            score += min(len(numbered_items) * 0.05, 0.3)

        # Bullet lists
        bullet_items = self._bullet_list_pattern.findall(content)
        if bullet_items:
            score += min(len(bullet_items) * 0.03, 0.2)

        # Headers (markdown style)
        headers = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
        if headers:
            score += min(len(headers) * 0.05, 0.2)

        # Tables
        tables = re.findall(r'\|.*\|', content)
        if tables:
            score += min(len(tables) * 0.05, 0.2)

        return min(score, 1.0)

    def _calculate_example_density(self, content: str) -> float:
        """Calculate example density score."""
        # Find sections (approximate by paragraphs)
        sections = [s.strip() for s in content.split('\n\n') if s.strip()]
        if not sections:
            return 0.0

        # Count example indicators
        example_indicators = [
            r'\bexample\b', r'\bfor instance\b', r'\be\.g\.\b', r'\bsuch as\b',
            r'\bdemonstrat\b', r'\bsample\b', r'\billustrat\b'
        ]

        example_count = 0
        for indicator in example_indicators:
            matches = re.findall(indicator, content, re.IGNORECASE)
            example_count += len(matches)

        # Also count code blocks as examples
        code_blocks = self._code_block_pattern.findall(content)
        example_count += len(code_blocks)

        # Calculate density
        density = example_count / len(sections)

        return min(density, 1.0)

    def _calculate_specificity(self, sentence: str) -> float:
        """Calculate specificity weight for a sentence."""
        score = 0.5  # Base score

        # Bonus for technical terms
        technical_terms = self._extract_technical_terms(sentence)
        if technical_terms:
            score += min(len(technical_terms) * 0.1, 0.3)

        # Bonus for numbers and measurements
        if re.search(r'\d+', sentence):
            score += 0.1

        # Bonus for specific verbs (actionable verbs)
        actionable_verbs_found = sum(1 for verb in self.ACTIONABLE_VERBS if verb in sentence)
        if actionable_verbs_found > 0:
            score += 0.1

        return min(score, 1.0)

    def _calculate_context_score(self, content: str, technical_terms: list[str]) -> float:
        """Calculate context appropriateness score for technical terms."""
        if not technical_terms:
            return 0.0

        context_scores = []

        for term in technical_terms:
            # Find sentences containing the term
            sentences_with_term = [
                s for s in re.split(r'[.!?]+', content)
                if term in s.lower()
            ]

            for sentence in sentences_with_term:
                # Check if other technical terms or related words appear nearby
                other_terms = [t for t in technical_terms if t != term and t in sentence.lower()]

                # Score based on co-occurrence with related terms
                if other_terms:
                    context_scores.append(1.0)
                else:
                    # Check for domain-related keywords
                    has_context = any(
                        keyword in sentence.lower()
                        for domain_terms in self.TECHNICAL_TERMS.values()
                        for keyword in domain_terms
                    )
                    context_scores.append(0.7 if has_context else 0.3)

        if not context_scores:
            return 0.5  # Neutral score if no context can be determined

        return float(np.mean(context_scores))
