import axios from 'axios'

export const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 120000, // Long timeout for heavy local model inference
})

// Demo API endpoints
export const demoApi = {
  chunkText: async (text: string, sourceName?: string) => {
    const res = await apiClient.post('/demo/text/chunk', { text, source_name: sourceName })
    return res.data
  },
  
  embedText: async (text: string) => {
    const res = await apiClient.post('/demo/text/embed', { text })
    return res.data
  },
  
  processImage: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post('/demo/image/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return res.data
  },
  
  processPdf: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post('/demo/pdf/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return res.data
  },
  
  processDocx: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post('/demo/docx/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return res.data
  },
  
  processAudio: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post('/demo/audio/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return res.data
  },
  
  getSystemReadiness: async () => {
    const res = await apiClient.get('/demo/system-readiness')
    return res.data
  },
  
  getAudioReadiness: async () => {
    const res = await apiClient.get('/demo/audio-readiness-check')
    return res.data
  },

  getLibrary: async () => {
    const res = await apiClient.get('/demo/library')
    return res.data
  },

  uploadToLibrary: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post('/demo/library/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return res.data
  },

  uploadTextNote: async (title: string, content: string) => {
    const formData = new FormData()
    formData.append('text_title', title)
    formData.append('text_content', content)
    const res = await apiClient.post('/demo/library/upload', formData)
    return res.data
  },

  deleteFromLibrary: async (id: string) => {
    const res = await apiClient.delete(`/demo/library/${id}`)
    return res.data
  },

  clearLibrary: async () => {
    const res = await apiClient.post('/demo/library/clear')
    return res.data
  }
}
