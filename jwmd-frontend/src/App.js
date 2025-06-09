import React, { useEffect, useState } from 'react';
import jsPDF from 'jspdf';

function App() {
  const [botellas, setBotellas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch botellas
  useEffect(() => {
    fetch('http://localhost:3000/botella') // Cambia URL si es necesario
      .then(res => {
        if (!res.ok) throw new Error('Error al cargar botellas');
        return res.json();
      })
      .then(data => {
        // Suponemos que data es array de botellas
        // Ordenamos por id descendente (más recientes primero)
        data.sort((a,b) => b.id.localeCompare(a.id));
        setBotellas(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      })
  }, []);

  // Separar botellas válidas e inválidas
  const botellasValidas = botellas.filter(b => b.estado === 1);
  const botellasInvalidas = botellas.filter(b => b.estado !== 1);

  // Exportar PDF usando jsPDF
  const exportPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text('Reporte de Botellas', 10, 10);

    let y = 20;
    doc.setFontSize(12);
    doc.text('Botellas Válidas (Empaquetadas):', 10, y);
    y += 10;

    botellasValidas.forEach((botella, i) => {
      doc.text(`${i+1}. ID: ${botella.id}`, 10, y);
      y += 7;
      doc.text(`   Ruta: ${botella.ruta}`, 10, y);
      y += 7;
      if (y > 280) { doc.addPage(); y = 10; }
    });

    y += 10;
    doc.text('Botellas Inválidas (con razón):', 10, y);
    y += 10;

    botellasInvalidas.forEach((botella, i) => {
      doc.text(`${i+1}. ID: ${botella.id}`, 10, y);
      y += 7;
      doc.text(`   Razón: ${botella.descripcion}`, 10, y);
      y += 7;
      if (y > 280) { doc.addPage(); y = 10; }
    });

    doc.save('botellas.pdf');
  };

  if (loading) return <p>Cargando botellas...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div style={{ maxWidth: 900, margin: 'auto', padding: 20, fontFamily: 'Arial' }}>
      <h1>Listado de Botellas</h1>
      <button onClick={exportPDF} style={{ marginBottom: 20, padding: '8px 12px', cursor: 'pointer' }}>
        Exportar a PDF
      </button>

      <h2>Botellas Válidas (Empaquetadas)</h2>
      {botellasValidas.length === 0 ? (
        <p>No hay botellas válidas.</p>
      ) : (
        <table border="1" cellPadding="5" cellSpacing="0" style={{width: '100%', marginBottom: 40}}>
          <thead>
            <tr>
              <th>ID</th>
              <th>Imagen</th>
              <th>Ruta</th>
              <th>Descripción</th>
            </tr>
          </thead>
          <tbody>
            {botellasValidas.map(b => (
              <tr key={b.id}>
                <td>{b.id}</td>
                <td>
                  <img src={b.imagen} alt="Botella" style={{height: 60}} />
                </td>
                <td>{b.ruta}</td>
                <td>{b.descripcion}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Botellas Inválidas</h2>
      {botellasInvalidas.length === 0 ? (
        <p>No hay botellas inválidas.</p>
      ) : (
        <table border="1" cellPadding="5" cellSpacing="0" style={{width: '100%'}}>
          <thead>
            <tr>
              <th>ID</th>
              <th>Imagen</th>
              <th>Razón</th>
            </tr>
          </thead>
          <tbody>
            {botellasInvalidas.map(b => (
              <tr key={b.id}>
                <td>{b.id}</td>
                <td>
                  <img src={b.imagen} alt="Botella" style={{height: 60}} />
                </td>
                <td>{b.descripcion}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;