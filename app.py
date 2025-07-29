import streamlit as st
from process import prepare_rag_inputs
from rag_chain import retrieve_chunks, build_prompt, generate_answer

st.set_page_config(page_title="RAG chain", layout="centered")
st.title("RAG")

if "index" not in st.session_state:
    with st.spinner("Reading"):
        chunks, embeddings, embed_model, index = prepare_rag_inputs("data/cc.pdf")
        st.session_state.chunks = chunks
        st.session_state.embed_model = embed_model
        st.session_state.index = index

query = st.text_input("Ask the document:")

if query:
    with st.spinner("thinking"):
        context = "\n\n".join(retrieve_chunks(query, st.session_state.embed_model, st.session_state.chunks, st.session_state.index))
        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

    st.markdown("Answer:")
    st.write(answer)

    with st.expander("Rtrieved Context"):
        st.write(context)
