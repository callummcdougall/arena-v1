import torch as t
import balanced_brackets_solutions

def test_get_inputs(get_inputs, model, data):

    module = model.layers[1].linear2

    expected = balanced_brackets_solutions.get_inputs(model, data, module)
    actual = get_inputs(model, data, module)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_inputs` passed.")

def test_get_outputs(get_outputs, model, data):

    module = model.layers[1].linear2

    expected = balanced_brackets_solutions.get_outputs(model, data, module)
    actual = get_outputs(model, data, module)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_outputs` passed.")

def test_get_out_by_head(get_out_by_head, model, data):

    layer = 2

    expected = balanced_brackets_solutions.get_out_by_head(model, data, layer)
    actual = get_out_by_head(model, data, layer)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_out_by_head` passed.")

def test_get_out_by_component(get_out_by_components, model, data):

    expected = balanced_brackets_solutions.get_out_by_components(model, data)
    actual = get_out_by_components(model, data)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_out_by_component` passed.")

def test_final_ln_fit(model, data, get_ln_fit):

    expected, exp_r2 = balanced_brackets_solutions.get_ln_fit(model, data, model.norm, 0)
    actual, act_r2 = get_ln_fit(model, data, model.norm, 0)

    t.testing.assert_close(t.tensor(actual.coef_), t.tensor(expected.coef_))
    t.testing.assert_close(t.tensor(actual.intercept_), t.tensor(expected.intercept_))
    t.testing.assert_close(act_r2, exp_r2)
    print("All tests in `test_final_ln_fit` passed.")

def test_pre_final_ln_dir(model, data, get_pre_final_ln_dir):

    expected = balanced_brackets_solutions.get_pre_final_ln_dir(model, data)
    actual = get_pre_final_ln_dir(model, data)
    similarity = t.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    t.testing.assert_close(similarity, 1.0)
    print("All tests in `test_pre_final_ln_dir` passed.")

def test_get_WV(model, get_WV):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        v = balanced_brackets_solutions.get_WV(model, layer, head)
        their_v = get_WV(model, layer, head)
        t.testing.assert_close(their_v, v)
    print("All tests in `test_get_WV` passed.")

def test_get_WO(model, get_WO):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        o = balanced_brackets_solutions.get_WO(model, layer, head)
        their_o = get_WO(model, layer, head)
        t.testing.assert_close(their_o, o)
    print("All tests in `test_get_WO` passed.")

def test_get_pre_20_dir(model, data, get_pre_20_dir):

    expected = balanced_brackets_solutions.get_pre_20_dir(model, data)
    actual = get_pre_20_dir(model, data)
    
    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_pre_20_dir` passed.")

def embedding_test(model, tokenizer, embedding_fn):

    open_encoding = balanced_brackets_solutions.embedding(model, tokenizer, "(")
    closed_encoding = balanced_brackets_solutions.embedding(model, tokenizer, ")")

    t.testing.assert_close(embedding_fn(model, tokenizer, "("), open_encoding)
    t.testing.assert_close(embedding_fn(model, tokenizer, ")"), closed_encoding)
    print("All tests in `embedding_test` passed.")

def qk_test(model, their_get_q_and_k):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        q, k = balanced_brackets_solutions.get_Q_and_K(model, layer, head)
        their_q, their_k = their_get_q_and_k(model, layer, head)
        t.testing.assert_close(their_q, q)
        t.testing.assert_close(their_k, k)
    print("All tests in `qk_test` passed.")

def test_qk_calc_termwise(model, tokenizer, their_get_q_and_k):

    embedding = model.encoder(tokenizer.tokenize(["()()()()"]).to(balanced_brackets_solutions.DEVICE)).squeeze()
    expected = balanced_brackets_solutions.qk_calc_termwise(model, 0, 0, embedding, embedding)
    actual = their_get_q_and_k(model, 0, 0, embedding, embedding)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_qk_calc_termwise` passed.")
