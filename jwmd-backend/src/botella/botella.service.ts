import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Botella } from './botella.entity';

@Injectable()
export class BotellaService {
    constructor(
        @InjectRepository(Botella)
        private BotellaRepository: Repository<Botella>
    ) {}


    //Encontrar todos
    async findAll(): Promise<Botella[]> {
        return this.BotellaRepository.find();
    }
    //Encontrar solo uno
    async findOne(id: string): Promise<Botella> {
        const botella = await this.BotellaRepository.findOne({ where: {id}});
        if (!botella) {
            //Si no encuentra, siempre va a dar la misma excepcion
            throw new NotFoundException('La botella no existe');
        }
        else {
            //Si si encuentra, siempre va a devolver la botella
            return botella;
        }
    }
    //Este metodo no es idempotente
    async create(objetobotella: Partial<Botella>): Promise<Botella> {
        const botella = this.BotellaRepository.create(objetobotella);
        return this.BotellaRepository.save(botella);
    }
    //Actualizacion si es idempotente
    async update(id: string, objetobotella: Partial<Botella>): Promise<Botella>{
        const botella = await this.findOne(id)
        if (!botella) {
            //Si no encuentra, siempre va a dar la misma excepcion
            throw new NotFoundException('La botella no existe');
        }
        else {
            //Si si encuentra, actualizamos la botella
            Object.assign(botella, objetobotella)
            return this.BotellaRepository.save(botella)
        }
    }
    //Eliminacion tambien es idempotente
    async delete(id: string): Promise<void> {
        const botella = await this.findOne(id);
        if (!botella) {
            throw new NotFoundException('La botella no existe');
        }
        else {
            await this.BotellaRepository.remove(botella)
        }
    }
}
