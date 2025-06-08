import { useEffect, useState } from 'react';
import type { Botella } from '../types/Botella';
import { getBotellas } from '../services/botellaService';
import BotellaListaValida from '../components/BotellaListaValida';
import BotellaListaInvalida from '../components/BotellaListaInvalida';

export default function ClasificacionBotellasPage() {
  const [validas, setValidas] = useState<Botella[]>([]);
  const [invalidas, setInvalidas] = useState<Botella[]>([]);

  useEffect(() => {
    getBotellas().then(res => {
      const todas = res.data;
      setValidas(todas.filter(b => b.clasificacion === 'valida'));
      setInvalidas(todas.filter(b => b.clasificacion === 'invalida'));
    });
  }, []);

  return (
    <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-6">
      <BotellaListaValida botellas={validas} />
      <BotellaListaInvalida botellas={invalidas} />
    </div>
  );
}
