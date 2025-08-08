import re
import os
import pypdf
import zlib
from collections import deque
from typing import List, Dict, Set, Tuple

# --- NLTK Imports ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print(
        "NLTK not installed. Advanced text processing (stop word removal, lemmatization, sentence tokenization) will be skipped.")
    print("Please install NLTK with: pip install nltk")

# --- Scikit-learn Imports ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not installed. Semantic similarity (TF-IDF, Cosine Similarity) will be skipped.")
    print("Please install scikit-learn with: pip install scikit-learn")

# --- Annoy Imports ---
try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    print(
        "Annoy not installed. Approximate Nearest Neighbors (ANN) optimization for section comparison will be skipped.")
    print("Please install Annoy with: pip install annoy")


class PlagiarismChecker:
    def __init__(self):
        print("PlagiarismChecker initialized for semantic similarity detection.")

        self.lemmatizer = None
        self.stop_words = set()
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                print("NLTK components (lemmatizer, stop words) initialized for PlagiarismChecker instance.")
            except Exception as e:
                print(f"Error initializing NLTK components within PlagiarismChecker (data missing?): {e}")
                print("NLTK advanced cleaning will be skipped for this PlagiarismChecker instance.")
                self.lemmatizer = None
                self.stop_words = set()

        self.vectorizer = None
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer()
            print("Document-level TF-IDF Vectorizer initialized.")
        else:
            print("Document-level TF-IDF Vectorizer not available (scikit-learn not installed).")

    def _extract_text_from_pdf(self, filepath: str) -> str:
        text = ""
        try:
            with open(filepath, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from {filepath}: {e}")
            return ""
        return text

    def _clean_and_normalize_text(self, text: str) -> str:
        """
        Cleans and normalizes the extracted text. Now includes NLTK-based processing if available.
        This version is optimized for internal sentence cleaning, so it won't print NLTK status.
        """
        text = text.lower()
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        text = re.sub(r'\([a-z\s]+(?:et al\.)?, \d{4}\)', '', text)

        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        text = re.sub(r'\s-\s', ' ', text)
        text = re.sub(r'^-|-$', '', text)
        text = re.sub(r'\s*-\s*', '-', text)

        text = re.sub(r'\s+', ' ', text).strip()

        if NLTK_AVAILABLE and self.lemmatizer and self.stop_words:
            words = word_tokenize(text)
            processed_words = []
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    processed_words.append(self.lemmatizer.lemmatize(word))
            text = " ".join(processed_words)
        return text

    def _segment_into_sentences(self, raw_text: str) -> List[Tuple[str, str]]:
        """
        Segments raw text into individual sentences using NLTK.
        Returns a list of (cleaned_sentence, original_sentence) tuples.
        """
        if not NLTK_AVAILABLE:
            print("NLTK not available for sentence segmentation. Using basic fallback.")
            original_sentences = re.split(r'(?<=[.!?])\s+', raw_text)
            return [(self._clean_and_normalize_text(s), s.strip()) for s in original_sentences if s.strip()]

        original_sentences = sent_tokenize(raw_text)

        sentence_pairs = []
        for original_s in original_sentences:
            cleaned_s = self._clean_and_normalize_text(original_s)
            if cleaned_s:
                sentence_pairs.append((cleaned_s, original_s.strip()))
        return sentence_pairs

    def _vectorize_texts(self, cleaned_texts: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Converts cleaned texts (full documents) into TF-IDF vectors.
        The vectorizer is fitted on the entire corpus of texts.
        """
        if not SKLEARN_AVAILABLE or not self.vectorizer:
            print("Document-level TF-IDF Vectorizer not available. Cannot vectorize texts.")
            return {}

        doc_ids = list(cleaned_texts.keys())
        texts_list = [cleaned_texts[doc_id] for doc_id in doc_ids]

        tfidf_matrix = self.vectorizer.fit_transform(texts_list)
        doc_vectors = {doc_ids[i]: tfidf_matrix[i].toarray()[0].tolist() for i in range(len(doc_ids))}
        return doc_vectors

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        """
        if not SKLEARN_AVAILABLE:
            return 0.0

        return cosine_similarity([vec1], [vec2])[0][0]

    def compare_documents_semantic(self, filepaths: List[str]) -> List[Dict]:
        """
        Orchestrates the semantic plagiarism detection process using TF-IDF and Cosine Similarity.
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn is not available. Cannot perform semantic comparison.")
            return []

        all_cleaned_texts: Dict[str, str] = {}

        print("\n--- STEP 1 & 2 (Combined): Extracting, Cleaning, and Normalizing Texts ---")
        for filepath in filepaths:
            doc_id = os.path.basename(filepath)
            print(f"\nProcessing document: {doc_id}")
            raw_text = self._extract_text_from_pdf(filepath)
            if not raw_text:
                print(f"Warning: No text extracted from {doc_id}. Skipping.")
                continue

            cleaned_text = self._clean_and_normalize_text(raw_text)
            if not cleaned_text:
                print(f"Warning: Text for {doc_id} became empty after cleaning. Skipping.")
                continue
            all_cleaned_texts[doc_id] = cleaned_text

            print("Extracted Raw Text (first 500 chars):")
            print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
            print(f"Total raw text length: {len(raw_text)} characters.")
            print("Cleaned and Normalized Text (first 500 chars):")
            print(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
            print(f"Total cleaned text length: {len(cleaned_text)} characters.")
            print(f"Finished processing text for {doc_id}.")

        if len(all_cleaned_texts) < 2:
            print("Not enough documents with valid text to compare.")
            return []

        print("\n--- STEP 3: Vectorizing All Documents (TF-IDF) ---")
        doc_vectors = self._vectorize_texts(all_cleaned_texts)
        if not doc_vectors:
            print("Failed to vectorize documents. Cannot proceed with comparison.")
            return []

        print(f"Vocabulary size (from TF-IDF): {len(self.vectorizer.vocabulary_)} unique words.")
        print(f"Vectorized {len(doc_vectors)} documents.")

        print("\n--- STEP 4: Calculating Pairwise Cosine Similarities (Document Level) ---")
        results = []
        doc_ids = list(doc_vectors.keys())

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc1_id = doc_ids[i]
                doc2_id = doc_ids[j]

                vec1 = doc_vectors[doc1_id]
                vec2 = doc_vectors[doc2_id]

                similarity = self.calculate_cosine_similarity(vec1, vec2)
                results.append({
                    'doc1': doc1_id,
                    'doc2': doc2_id,
                    'similarity': similarity
                })
        return results

    def _aggregate_matching_sections(self, matching_sections: List[Dict]) -> List[Dict]:
        """
        Aggregates consecutive matching sentences into larger blocks.
        Assumes matching_sections is sorted by doc1_sentence_index.
        """
        if not matching_sections:
            return []

        # Sort by doc1_sentence_index and then by doc2_sentence_index to ensure proper sequence
        sorted_matches = sorted(matching_sections, key=lambda x: (x['doc1_sentence_index'], x['doc2_sentence_index']))

        aggregated_blocks = []
        current_block = None

        for match in sorted_matches:
            doc1_idx = match['doc1_sentence_index']
            doc2_idx = match['doc2_sentence_index']
            doc1_orig_s = match['doc1_original_sentence']
            doc2_orig_s = match['doc2_original_sentence']
            similarity = match['similarity']

            if current_block is None:
                current_block = {
                    'doc1_start_index': doc1_idx,
                    'doc1_end_index': doc1_idx,
                    'doc2_start_index': doc2_idx,
                    'doc2_end_index': doc2_idx,
                    'doc1_sentences': [doc1_orig_s],
                    'doc2_sentences': [doc2_orig_s],
                    'avg_similarity': similarity,
                    'count': 1
                }
            else:
                is_consecutive_doc1 = (doc1_idx == current_block['doc1_end_index'] + 1)
                is_consecutive_doc2 = (doc2_idx == current_block['doc2_end_index'] + 1)

                if is_consecutive_doc1 and is_consecutive_doc2:
                    current_block['doc1_end_index'] = doc1_idx
                    current_block['doc2_end_index'] = doc2_idx
                    current_block['doc1_sentences'].append(doc1_orig_s)
                    current_block['doc2_sentences'].append(doc2_orig_s)
                    current_block['avg_similarity'] = (current_block['avg_similarity'] * current_block[
                        'count'] + similarity) / (current_block['count'] + 1)
                    current_block['count'] += 1
                else:
                    aggregated_blocks.append(current_block)
                    current_block = {
                        'doc1_start_index': doc1_idx,
                        'doc1_end_index': doc1_idx,
                        'doc2_start_index': doc2_idx,
                        'doc2_end_index': doc2_idx,
                        'doc1_sentences': [doc1_orig_s],
                        'doc2_sentences': [doc2_orig_s],
                        'avg_similarity': similarity,
                        'count': 1
                    }
        if current_block:
            aggregated_blocks.append(current_block)

        formatted_blocks = []
        for block in aggregated_blocks:
            formatted_blocks.append({
                'doc1_text': " ".join(block['doc1_sentences']),
                'doc2_text': " ".join(block['doc2_sentences']),
                'avg_similarity': block['avg_similarity'],
                'doc1_indices': (block['doc1_start_index'], block['doc1_end_index']),
                'doc2_indices': (block['doc2_start_index'], block['doc2_end_index']),
                'num_sentences': block['count']
            })
        return formatted_blocks

    def find_plagiarized_sections_semantic(self, doc1_filepath: str, doc2_filepath: str,
                                           similarity_threshold: float = 0.75, num_ann_trees: int = 10,
                                           num_ann_neighbors: int = 50) -> List[Dict]:
        """
        Identifies specific semantically similar sections (sentences) between two documents
        and aggregates them into blocks, using Annoy for performance optimization.

        Args:
            doc1_filepath (str): Path to the first document.
            doc2_filepath (str): Path to the second document.
            similarity_threshold (float): Cosine similarity threshold for a sentence pair match.
            num_ann_trees (int): Number of trees to build in the Annoy index (more trees = higher accuracy, slower build).
            num_ann_neighbors (int): Number of nearest neighbors to retrieve from Annoy for each query.

        Returns:
            List[Dict]: A list of dictionaries, each describing an aggregated matching block.
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn is not available. Cannot identify specific sections.")
            return []
        if not ANNOY_AVAILABLE:
            print("Annoy is not available. Performing full pairwise sentence comparison (slower).")
            # Fallback to original O(N*M) comparison if Annoy is not available
            return self._find_plagiarized_sections_full_comparison(doc1_filepath, doc2_filepath, similarity_threshold)

        doc1_id = os.path.basename(doc1_filepath)
        doc2_id = os.path.basename(doc2_filepath)
        print(
            f"\n{'=' * 10} Identifying Plagiarized Sections between {doc1_id} and {doc2_id} (ANN Optimized) {'=' * 10}")

        # --- Process Document 1 ---
        raw_text1 = self._extract_text_from_pdf(doc1_filepath)
        if not raw_text1:
            print(f"Warning: No text extracted from {doc1_id}. Cannot compare sections.")
            return []
        sentence_pairs1 = self._segment_into_sentences(raw_text1)
        sentences1_cleaned = [pair[0] for pair in sentence_pairs1]
        sentences1_original = [pair[1] for pair in sentence_pairs1]
        print(f"Document 1 ({doc1_id}): {len(sentences1_cleaned)} cleaned sentences.")

        # --- Process Document 2 ---
        raw_text2 = self._extract_text_from_pdf(doc2_filepath)
        if not raw_text2:
            print(f"Warning: No text extracted from {doc2_id}. Cannot compare sections.")
            return []
        sentence_pairs2 = self._segment_into_sentences(raw_text2)
        sentences2_cleaned = [pair[0] for pair in sentence_pairs2]
        sentences2_original = [pair[1] for pair in sentence_pairs2]
        print(f"Document 2 ({doc2_id}): {len(sentences2_cleaned)} cleaned sentences.")

        if not sentences1_cleaned or not sentences2_cleaned:
            print("One or both documents have no valid sentences after processing. Cannot compare sections.")
            return []

        # --- Vectorize all sentences from both documents for this specific comparison ---
        # A new vectorizer is created here to ensure its vocabulary is built only from these two documents' sentences.
        sentence_vectorizer = TfidfVectorizer()
        all_sentences_cleaned = sentences1_cleaned + sentences2_cleaned

        if not all_sentences_cleaned:
            print("No valid sentences to vectorize after cleaning. Cannot compare sections.")
            return []

        try:
            sentence_tfidf_matrix = sentence_vectorizer.fit_transform(all_sentences_cleaned)
        except ValueError as e:
            print(f"Error fitting TF-IDF vectorizer on sentences: {e}")
            print("This might happen if all sentences are too short or contain only stop words/punctuation.")
            return []

        vectors1 = sentence_tfidf_matrix[:len(sentences1_cleaned)]
        vectors2 = sentence_tfidf_matrix[len(sentences1_cleaned):]

        # --- Annoy Index Building for Performance ---
        # Get the dimensionality of the TF-IDF vectors
        vector_dim = vectors2.shape[1]
        if vector_dim == 0:
            print("Warning: Sentence vectors have zero dimensions. Cannot build Annoy index.")
            return []

        annoy_index = AnnoyIndex(vector_dim, 'angular')  # 'angular' for cosine similarity
        for i, vec in enumerate(vectors2):
            annoy_index.add_item(i, vec.toarray()[0])  # Annoy needs dense vectors
        annoy_index.build(num_ann_trees)  # Build the index with specified number of trees
        print(f"Annoy index built for {len(sentences2_cleaned)} sentences from {doc2_id} with {num_ann_trees} trees.")

        raw_matching_sections = []
        print(f"\nQuerying Annoy index for similar sentences (threshold: {similarity_threshold:.0%})...")

        for i, s1_vec in enumerate(vectors1):
            # Find num_ann_neighbors nearest neighbors in doc2 for current sentence from doc1
            # -1 for search_k means it will inspect all nodes, 0 for default (faster)
            nearest_indices, distances = annoy_index.get_nns_by_vector(s1_vec.toarray()[0], num_ann_neighbors,
                                                                       include_distances=True)

            for j_idx_in_nearest_list, s2_idx in enumerate(nearest_indices):
                # Annoy returns angular distance, convert to cosine similarity
                # angular distance = sqrt(2 * (1 - cosine_similarity))
                # cosine_similarity = 1 - (angular_distance^2 / 2)
                # Note: Annoy's distance is Euclidean distance in a transformed space for angular.
                # It's often simpler to just calculate the exact cosine_similarity between the original vectors
                # for the candidates found by Annoy.

                # Calculate exact cosine similarity for the candidate pair
                exact_similarity = cosine_similarity(s1_vec, vectors2[s2_idx])[0][0]

                if exact_similarity >= similarity_threshold:
                    raw_matching_sections.append({
                        'doc1_sentence_index': i,
                        'doc1_original_sentence': sentences1_original[i],
                        'doc2_sentence_index': s2_idx,
                        'doc2_original_sentence': sentences2_original[s2_idx],
                        'similarity': exact_similarity
                    })

        print(f"Found {len(raw_matching_sections)} raw matching sentence pairs above threshold.")

        aggregated_blocks = self._aggregate_matching_sections(raw_matching_sections)
        print(f"Aggregated into {len(aggregated_blocks)} plagiarism blocks.")

        return aggregated_blocks

    # --- Fallback method for find_plagiarized_sections_semantic if Annoy is not available ---
    def _find_plagiarized_sections_full_comparison(self, doc1_filepath: str, doc2_filepath: str,
                                                   similarity_threshold: float = 0.75) -> List[Dict]:
        """
        Fallback method for identifying specific sections using full pairwise comparison (slower).
        """
        doc1_id = os.path.basename(doc1_filepath)
        doc2_id = os.path.basename(doc2_filepath)
        print(
            f"\n{'=' * 10} Identifying Plagiarized Sections between {doc1_id} and {doc2_id} (Full Comparison) {'=' * 10}")

        raw_text1 = self._extract_text_from_pdf(doc1_filepath)
        sentence_pairs1 = self._segment_into_sentences(raw_text1)
        sentences1_cleaned = [pair[0] for pair in sentence_pairs1]
        sentences1_original = [pair[1] for pair in sentence_pairs1]

        raw_text2 = self._extract_text_from_pdf(doc2_filepath)
        sentence_pairs2 = self._segment_into_sentences(raw_text2)
        sentences2_cleaned = [pair[0] for pair in sentence_pairs2]
        sentences2_original = [pair[1] for pair in sentence_pairs2]

        if not sentences1_cleaned or not sentences2_cleaned:
            print("One or both documents have no valid sentences after processing. Cannot compare sections.")
            return []

        sentence_vectorizer = TfidfVectorizer()
        all_sentences_cleaned = sentences1_cleaned + sentences2_cleaned
        if not all_sentences_cleaned:
            print("No valid sentences to vectorize after cleaning. Cannot compare sections.")
            return []
        try:
            sentence_tfidf_matrix = sentence_vectorizer.fit_transform(all_sentences_cleaned)
        except ValueError as e:
            print(f"Error fitting TF-IDF vectorizer on sentences: {e}")
            return []

        vectors1 = sentence_tfidf_matrix[:len(sentences1_cleaned)]
        vectors2 = sentence_tfidf_matrix[len(sentences1_cleaned):]

        raw_matching_sections = []
        print(
            f"\nComparing {len(sentences1_cleaned)} sentences from {doc1_id} with {len(sentences2_cleaned)} sentences from {doc2_id} (threshold: {similarity_threshold:.0%}, full pairwise)...")

        for i, s1_vec in enumerate(vectors1):
            for j, s2_vec in enumerate(vectors2):
                similarity = cosine_similarity(s1_vec, s2_vec)[0][0]
                if similarity >= similarity_threshold:
                    raw_matching_sections.append({
                        'doc1_sentence_index': i,
                        'doc1_original_sentence': sentences1_original[i],
                        'doc2_sentence_index': j,
                        'doc2_original_sentence': sentences2_original[j],
                        'similarity': similarity
                    })
        print(f"Found {len(raw_matching_sections)} raw matching sentence pairs above threshold.")
        aggregated_blocks = self._aggregate_matching_sections(raw_matching_sections)
        print(f"Aggregated into {len(aggregated_blocks)} plagiarism blocks.")
        return aggregated_blocks


# --- Test Functions (unchanged for create_dummy_pdf) ---
def create_dummy_pdf(filepath: str, content: str):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(filepath, pagesize=letter)
        textobject = c.beginText()
        textobject.setTextOrigin(50, 750)
        lines = content.split('\n')
        y_position = 750
        for line in lines:
            if y_position < 50:
                c.showPage()
                y_position = 750
            textobject.textLine(line)
            y_position -= 12
        c.drawText(textobject)
        c.save()
    except ImportError:
        print("ReportLab not installed. Cannot create dummy PDF. Please install with 'pip install reportlab'")
        print("You'll need to manually create a PDF or provide an existing one for testing.")
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")


# --- Main execution for testing ---
if __name__ == "__main__":

    # --- Define contents for test papers ---
    content_paper1 = """
    Abstract
    This is a sample academic paper to test the plagiarism checker.
    It includes some common words, numbers like 123, and special characters.
    The quick brown fox jumps over the lazy dog.
    This sentence is repeated for testing.
    According to Smith (2020), this is a significant finding.
    Another sentence with the same idea.
    This research was supported by grants.

    Introduction
    The field of natural language processing has seen significant advancements in recent years.
    Plagiarism detection remains a challenging task, especially with the proliferation of digital content.
    Our approach leverages traditional techniques combined with modern insights.
    The methods described here are designed to be robust against minor alterations.

    Methodology
    We employed a robust text extraction pipeline. Data cleaning involved several stages.
    Shingling was performed using k-grams of varying lengths.
    Hashing functions were selected for their performance and collision resistance.
    This detailed methodology ensures reproducibility of our results.

    Conclusion
    In conclusion, the proposed system demonstrates promising capabilities for identifying text similarity.
    Future work will involve integrating more advanced linguistic processing and exploring neural network approaches.
    This paper contributes to the ongoing discussion on academic integrity.
    """

    # Modified content_paper2 to have more clear sentence-level overlaps
    # Added a consecutive block of similar sentences
    content_paper2 = """
    Abstract
    This is a second academic paper. It is designed to have some overlap with the first paper.
    It includes some common words and slightly different phrasing.
    The quick brown fox jumps over the lazy dog. This sentence is repeated for testing.
    This research was supported by various grants.
    Plagiarism detection remains a challenging task, especially with the proliferation of digital content.
    Our approach leverages traditional techniques combined with modern insights.
    Future work will involve integrating more advanced linguistic processing and exploring neural network approaches.
    In conclusion, the proposed system demonstrates promising capabilities for identifying text similarity.
    Future work will involve integrating more advanced linguistic processing and exploring neural network approaches.
    This paper contributes to the ongoing discussion on academic integrity.
    """

    content_paper3 = """
    This is a completely different paper with no similar content. It should have low similarity.
    It discusses the history of space exploration and the discovery of new exoplanets.
    The advancements in telescope technology have revolutionized our understanding of the universe.
    This document focuses on astrophysics rather than language processing.
    """

    pdf_files_to_check = {
        "test_paper_1.pdf": content_paper1,
        "test_paper_2.pdf": content_paper2,
        "test_paper_3.pdf": content_paper3
    }

    # --- Create dummy PDF files for testing ---
    print("\n--- Creating dummy PDF files for testing ---")
    for filename, content in pdf_files_to_check.items():
        create_dummy_pdf(filename, content)
        print(f"Created {filename}")

    # --- Initialize the Plagiarism Checker ---
    print("\n--- Initializing Plagiarism Checker ---")
    checker = PlagiarismChecker()

    # --- Process and Compare documents using Semantic approach (Document Level) ---
    print("\n--- Starting Document-Level Semantic Plagiarism Detection Process ---")
    document_level_comparison_results = checker.compare_documents_semantic(list(pdf_files_to_check.keys()))

    # --- Report Document-Level results ---
    if document_level_comparison_results:
        print("\n--- Document-Level Semantic Plagiarism Detection Results Summary ---")
        sorted_results = sorted(document_level_comparison_results, key=lambda x: x['similarity'], reverse=True)
        for result in sorted_results:
            print(f"Similarity between '{result['doc1']}' and '{result['doc2']}': {result['similarity']:.2%}")
            if result['similarity'] >= 0.40:  # Lowered threshold from 0.50 to 0.40
                print("  --> High semantic similarity detected! Potential plagiarism.")
                # Automatically trigger section identification for highly similar pairs

                # Call the new function to find and aggregate sections
                plagiarism_blocks = checker.find_plagiarized_sections_semantic(
                    result['doc1'], result['doc2'], similarity_threshold=0.75  # Higher threshold for specific matches
                )

                if plagiarism_blocks:
                    print("\n    --- Detailed Plagiarism Blocks ---")
                    sorted_blocks = sorted(plagiarism_blocks, key=lambda x: x['doc1_indices'][0])
                    for block in sorted_blocks:
                        print(f"      Block Similarity (Avg): {block['avg_similarity']:.2%}")
                        print(f"      Sentences in Block: {block['num_sentences']}")
                        print(f"      Doc1 Text (Original):")
                        print(f"        '{block['doc1_text']}'")
                        print(f"      Doc2 Text (Original):")
                        print(f"        '{block['doc2_text']}'")
                        print("-" * 80)
                else:
                    print("    No specific highly similar sentences found at the section-level threshold.")
            elif result['similarity'] >= 0.20:
                print("  --> Moderate semantic similarity. Worth reviewing.")
            else:
                print("  --> Low semantic similarity.")
    else:
        print("No document-level comparisons performed or no results found.")

    print("\n--- End of Plagiarism Checker Report ---")